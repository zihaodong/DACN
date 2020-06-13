import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
import os
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from collections import defaultdict
from tensorboardX import SummaryWriter
from tqdm import tqdm
import shutil
import sys
sys.path.append('.')
from Utils import utils
from active_spline import ActiveSplineTorch
from DataProvider import cityscape_from_eclipse_active_spline
from Models.GNN import poly_gnn
from Evaluation import losses, metrics
#add by dzh 7.18
from contours import ContourBox
#from Evaluation import WHD_losses
import torch.nn.functional as F
import Evaluation.focal_lossv2 as focal
#import Evaluation.focal_loss as focal
from Evaluation.DiffRender_spline .py2d import diff_render_spline_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-7


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()

    return args


def get_data_loaders(opts, DataProvider):
    print 'Building dataloaders'

    if opts['debug']:
        opts['dataset']['train']['batch_size'] = 2
        opts['dataset']['train_val']['batch_size'] = 2

        opts['dataset']['train']['class_filter'] = ['car']
        do_shuffle = False
    else:
        do_shuffle = True

    opts['dataset']['train']['ext_points_pert'] = opts['ext_points_pert']
    opts['dataset']['train_val']['ext_points_pert'] = 0
    opts['dataset']['train']['ext_points'] = opts['ext_points']
    opts['dataset']['train_val']['ext_points'] = opts['ext_points']
    opts['dataset']['train']['p_num'] = opts['p_num']
    opts['dataset']['train_val']['p_num'] = opts['p_num']
    opts['dataset']['train']['cp_num'] = opts['cp_num']
    opts['dataset']['train_val']['cp_num'] = opts['cp_num']

    dataset_train = DataProvider(split='train', opts=opts['dataset']['train'], debug=opts['debug'])
    dataset_val = DataProvider(split='train_val', opts=opts['dataset']['train_val'], debug=opts['debug'])

    train_loader = DataLoader(dataset_train, batch_size=opts['dataset']['train']['batch_size'],
        shuffle=do_shuffle, num_workers=opts['dataset']['train']['num_workers'], collate_fn=cityscape_from_eclipse_active_spline.collate_fn)

    val_loader = DataLoader(dataset_val, batch_size=opts['dataset']['train_val']['batch_size'],
        shuffle=False, num_workers=opts['dataset']['train_val']['num_workers'], collate_fn=cityscape_from_eclipse_active_spline.collate_fn)

    return train_loader, val_loader


class Trainer(object):
    def __init__(self, args):
        self.global_step = 0
        self.epoch = 0

        self.opts = json.load(open(args.exp, 'r'))

        if 'test' in self.opts['exp_dir'] and os.path.exists(self.opts['exp_dir']):
            shutil.rmtree(os.path.join(self.opts['exp_dir']))

        utils.create_folder(os.path.join(self.opts['exp_dir'], 'checkpoints'))

        # Copy experiment file
        os.system('cp %s %s' % (args.exp, self.opts['exp_dir']))

        self.writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train'))
        self.val_writer = SummaryWriter(os.path.join(self.opts['exp_dir'], 'logs', 'train_val'))

        self.spline = ActiveSplineTorch(self.opts['cp_num'], self.opts[u'p_num'], device=device, alpha=self.opts['spline_alpha'])
        self.train_loader, self.val_loader = get_data_loaders(self.opts, cityscape_from_eclipse_active_spline.DataProvider)

        self.model = poly_gnn.PolyGNN(state_dim=self.opts['state_dim'],
                                      n_adj=self.opts['n_adj'],
                                      cnn_feature_grids=self.opts['cnn_feature_grids'],
                                      coarse_to_fine_steps=self.opts['coarse_to_fine_steps'],
                                      get_point_annotation=self.opts['get_point_annotation'],
                                      ).to(device)

        if 'xe_initializer' in self.opts.keys():
            self.model.reload(self.opts['xe_initializer'])
        elif 'encoder_reload' in self.opts.keys():

            self.model.encoder.reload(self.opts['encoder_reload'])


        # OPTIMIZER
        no_wd = []
        wd = []
        print 'Weight Decay applied to: '

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                # No optimization for frozen params
                continue

            if 'bn' in name or 'bias' in name:
                no_wd.append(p)
            else:
                wd.append(p)
                # print name,

        # Allow individual options
        self.optimizer = optim.Adam(
            [
                {'params': no_wd, 'weight_decay': 0.0},
                {'params': wd}
            ],
            lr=self.opts['lr'],
            weight_decay=self.opts['weight_decay'],
            amsgrad=False)

        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opts['lr_decay'],
                                                  gamma=0.1)

        if args.resume is not None:
            self.resume(args.resume)

        if self.opts['debug']:
            print "********************* we are in debug mode *********************"


    def save_checkpoint(self, epoch):
        save_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_decay': self.lr_decay.state_dict()
        }

        save_name = os.path.join(self.opts['exp_dir'], 'checkpoints', 'epoch%d_step%d.pth' \
                                 % (epoch, self.global_step))
        torch.save(save_state, save_name)
        print 'Saved model'


    def resume(self, path):
        self.model.reload(path)
        save_state = torch.load(path)
        self.global_step = save_state['global_step']
        self.epoch = save_state['epoch']
        self.optimizer.load_state_dict(save_state['optimizer'])
        self.lr_decay.load_state_dict(save_state['lr_decay'])


    def loop(self):
        for epoch in range(self.epoch, self.opts['max_epochs']):
            if not self.opts['debug']:
                self.save_checkpoint(epoch)
                self.lr_decay.step()
                print 'LR is now: ', self.optimizer.param_groups[0]['lr']
            self.train(epoch)


    def train(self, epoch):
        print 'Starting training'
        self.model.train()
        accum = defaultdict(float)
        #loss_loc = WHD_losses.WeightedHausdorffDistance(resized_height=self.opts['p_num'],resized_width=self.opts['p_num'],
        #                                                return_2_terms=True,
        #                                                device=device)
        #loss_loc = WHD_losses.AveragedHausdorffLoss()
        loss_loc = losses.WeightedHausdorffDistance(resized_height=224,
                                            resized_width=224,
                                            return_2_terms=True,
                                            device=device)
        # focalloss = focal.FocalLoss(None,None,None,'mean')
        focalloss = focal.FocalLoss()
        # To accumulate stats for printing
        for step, data in enumerate(self.train_loader):
            if len(data['img']) == 1:
                continue
            if self.opts['get_point_annotation']:
                img = data['img'].to(device)
                annotation = data['annotation_prior'].to(device).unsqueeze(1)
                img = torch.cat([img, annotation], 1)
            else:
                img = data['img'].to(device)
            self.optimizer.zero_grad()
            if self.global_step % self.opts['val_freq'] == 0 and not self.opts['debug']:
                self.validate()
                self.save_checkpoint(epoch)
            output = self.model.forward(img, data['fwd_poly'])
            loss_sum = 0
            pred_cps = output['pred_polys'][-1]
            pred_polys = self.spline.sample_point(pred_cps)
            # print(pred_polys.shape)
            # print(output['vertex_logits'].shape)
            gt_right_order, poly_mathcing_loss_sum = losses.poly_mathcing_loss(self.opts['p_num'],
                                                  pred_polys,
                                                  data['gt_poly'].to(device),
                                                  loss_type=self.opts['loss_type'])
            # add by dzh contour refine
            ## Initializing Contour Box
            level_set_config_dict = {
                'step_ckpts': [50],
                'lambda_': 0.0,
                'alpha': 1,
                'smoothing': 1,
                'render_radius': -1,
                'is_gt_semantic': True,
                'method': 'MLS',
                'balloon': 1,
                'threshold': 0.99,
                'merge_weight': 0.5
            }
            cbox = ContourBox.LevelSetAlignment(n_workers=1,
                                                fn_post_process_callback=None,
                                                config=level_set_config_dict)
            # print('-------------shape--------------------')  
            output_contour, _ = cbox({'seg': np.expand_dims(data['edge_mask'], 0), 'bdry': None},
                 np.expand_dims(output['edge_logits'].view(data['edge_mask'].shape).cpu().detach().numpy(), 0))
            masks_step = output_contour[0, :, 0, :, :]
            #--------add by dzh 7.18
            edge_annotation_loss = 0
            curr_fp_edge_loss = losses.fp_edge_loss(torch.from_numpy(masks_step).to(device),#self.opts['fp_weight'] * losses.fp_edge_loss(torch.from_numpy(masks_step).to(device),
                                                                             output['edge_logits'])#data['edge_mask'] torch.from_numpy(masks_step)
            edge_annotation_loss +=  curr_fp_edge_loss
            tt = []
            #pred_poly_mask = np.zeros((36, 36), np.float32)
            for i in range(pred_polys.shape[0]):
                pred_poly_mask = np.zeros((224, 224), dtype=np.float32)
                ff = np.floor(pred_polys[i].detach().cpu().numpy() * 36).astype(np.int32)

                if not isinstance(ff, list):
                    ff = [ff]
                for p in ff:
                    pred_poly_mask = utils.draw_poly(pred_poly_mask, p)
                #ff=utils.poly01_to_poly0g(pred_polys[i].detach().cpu().numpy(), 35)
                # pred_poly_mask = utils.get_vertices_mask_36(ff, pred_poly_mask)
                tt.append(pred_poly_mask)
            tt1 = np.array(tt,dtype=np.float32)
            pred_poly_mask11 = torch.from_numpy(tt1).cuda()
            ll1 = pred_poly_mask11
            #ll1 = output['vertex_logits'].view(output['vertex_logits'].shape[0],28,28)
            jjj = []
            for tt in range(ll1.shape[0]):
                jjj.append([224,224])
            #jjj = [[28,28],[28,28],[28,28],[28,28],[28,28],[28,28],[28,28],[28,28],[28,28],[28,28],[28,28],[28,28],[28,28],[28,28],[28,28],[28,28]]
            # print(data['poly_mask'].shape)
            kk = []
            poly_mask_ori = data['poly_mask']
            for hh in range(ll1.shape[0]):
                zzz = torch.FloatTensor(poly_mask_ori[hh].astype(np.float32)).cuda()
                #zzz = torch.FloatTensor(data['gt_orig_poly'][hh]).cuda()
                kk.append(zzz)
            #zzz = torch.from_numpy(data['gt_orig_poly'][0])
            # print(ll1.shape)
            # print(kk.shape)
            #ll1,kk
            term1, term2 = loss_loc.forward(ll1,
                                            kk,
                                            torch.FloatTensor(np.array(jjj,dtype=np.float32)).cuda())
            #fp_vertex_loss = self.opts['fp_weight'] * (term1+term2)     
            fp_vertex_loss = 0.1*(term1+term2) +poly_mathcing_loss_sum
            #fp_vertex_loss = poly_mathcing_loss_sum + self.opts['fp_weight']* 0.1 * (term1+term2)
            loss_sum += fp_vertex_loss
            loss_sum += edge_annotation_loss# + self.opts['fp_weight'] * (term1+term2)
            ################iou loss function#################
            #preds= pred_polys.detach().data.cpu().numpy()
            #iou_loss = 0
            #orig_poly = data['orig_poly']

            #for i in range(preds.shape[0]):
            #    curr_pred_poly = np.floor(preds[i] * 224).astype(np.int32)
            #    curr_gt_poly = np.floor(orig_poly[i] * 224).astype(np.int32)
            #    cur_iou, masks = metrics.iou_from_poly(np.array(curr_pred_poly, dtype=np.int32),
            #                                                    np.array(curr_gt_poly, dtype=np.int32),
            #                                                    224, 224)    
            #    iou_loss += cur_iou
            #iou_loss = -iou_loss / preds.shape[0]
            #loss_sum += 0.1 * iou_loss
            ################iou loss function#################
            with torch.no_grad():
                iou = 0
                gt_mask_0 = []
                pred_mask_0 = []
                orig_poly = data['orig_poly']
                preds= pred_polys.detach().data.cpu().numpy()
                # iou_filter = []
                for i in range(preds.shape[0]):
                    curr_pred_poly = np.floor(preds[i] * 224).astype(np.int32)
                    curr_gt_poly = np.floor(orig_poly[i] * 224).astype(np.int32)
                    cur_iou, masks = metrics.iou_from_poly(np.array(curr_pred_poly, dtype=np.int32),
                                                           np.array(curr_gt_poly, dtype=np.int32),
                                                           224,
                                                           224)
                    gt_mask_0.append(masks[1])
                    pred_mask_0.append(masks[0])
            gt_mask_1 = torch.from_numpy(np.array(gt_mask_0)).to(device).float()
            pred_mask_1 = torch.from_numpy(np.array(pred_mask_0)).to(device).float()    
            # mask_loss = focalloss(pred_mask_1, gt_mask_1)
            # mask_loss = losses.class_balanced_cross_entropy_loss(pred_mask_1, gt_mask_1)
            # pred111=pred_mask_1.view(pred_mask_1.shape[0],1,224,224)
            #mask_loss = 100 * focalloss((pred_mask_1/255), (gt_mask_1/255))    
            mask_loss =  torch.sum(torch.abs(gt_mask_1/250 - pred_mask_1/250))
            loss_sum += torch.mean(mask_loss)
            #         # iou_filter.append(1 if cur_iou>self.opts['iou_filter'] else 0)
            #         iou += cur_iou
            # iou = iou / preds.shape[0]
            # # iou_filter = np.array(iou_filter)
            # # iou_filter = torch.from_numpy(iou_filter).to(device).float()

            # loss_sum += (-iou)
            # if self.opts['iou_filter']>0:
            #     loss_sum = (loss_sum + (1-iou)) * iou_filter 

            # loss_sum = torch.mean(loss_sum)
            loss_sum.backward()
            if 'grad_clip' in self.opts.keys():
                nn.utils.clip_grad_norm_(self.model.parameters(), self.opts['grad_clip'])
            self.optimizer.step()
            preds= pred_polys.detach().data.cpu().numpy()
            with torch.no_grad():
                # Get IoU
                iou = 0
                orig_poly = data['orig_poly']

                for i in range(preds.shape[0]):
                    curr_pred_poly = np.floor(preds[i] * 224).astype(np.int32)
                    curr_gt_poly = np.floor(orig_poly[i] * 224).astype(np.int32)


                    cur_iou, masks = metrics.iou_from_poly(np.array(curr_pred_poly, dtype=np.int32),
                                                           np.array(curr_gt_poly, dtype=np.int32),
                                                           224,
                                                           224)
                    iou += cur_iou
                iou = iou / preds.shape[0]
                accum['loss'] += float(loss_sum.item())
                accum['iou'] += iou
                accum['length'] += 1
                if self.opts['edge_loss']:
                    accum['edge_annotation_loss'] += float(edge_annotation_loss.item())
                print(
                    "[%s] Epoch: %d, Step: %d, Polygon Loss: %f,  IOU: %f" \
                    % (str(datetime.now()), epoch, self.global_step, accum['loss'] / accum['length'], accum['iou'] / accum['length']))
                if step % self.opts['print_freq'] == 0:
                    # Mean of accumulated values
                    for k in accum.keys():
                        if k == 'length':
                            continue
                        accum[k] /= accum['length']

                    # Add summaries
                    masks = np.expand_dims(masks, -1).astype(np.uint8)  # Add a channel dimension
                    #print(masks.shape)
                    masks = np.tile(masks, [1, 1, 1, 3])  # Make [2, H, W, 3]
                    img = (data['img'].cpu().numpy()[-1, ...] * 255).astype(np.uint8)
                    img = np.transpose(img, [1, 2, 0])  # Make [H, W, 3]
                    self.writer.add_image('pred_mask', masks[0], self.global_step)
                    self.writer.add_image('gt_mask', masks[1], self.global_step)
                    self.writer.add_image('image', img, self.global_step)
                    self.writer.add_image('edge_acm_gt',np.tile(np.expand_dims(masks_step[preds.shape[0]-1],axis=-1).astype(np.uint8),[1, 1, 3]),self.global_step)
                    #self.writer.add_image('ori_GT',
                    pred_edge_mask = np.tile(np.expand_dims(output['edge_logits'].cpu().numpy()[preds.shape[0]-1]*255,axis=-1).astype(np.uint8),[1, 1, 3]).reshape(28,28,3)
                    #print(pred_edge_mask.shape)
                    self.writer.add_image('pred_edge',pred_edge_mask, self.global_step)
                    for k in accum.keys():
                        if k == 'length':
                            continue
                        self.writer.add_scalar(k, accum[k], self.global_step)
                    print(
                    "[%s] Epoch: %d, Step: %d, Polygon Loss: %f,  IOU: %f" \
                    % (str(datetime.now()), epoch, self.global_step, accum['loss'], accum['iou']))

                    accum = defaultdict(float)

            del (output, masks, pred_polys, preds, loss_sum)
            self.global_step += 1


    def validate(self):
        print 'Validating'
        self.model.eval()
        # Leave LSTM in train mode

        with torch.no_grad():
            ious = []
            for step, data in enumerate(tqdm(self.val_loader)):
                if len(data['orig_poly']) == 1:
                    continue
                if self.opts['get_point_annotation']:
                    img = data['img'].to(device)
                    annotation = data['annotation_prior'].to(device).unsqueeze(1)
                    img = torch.cat([img, annotation], 1)
                else:
                    img = data['img'].to(device)
                output = self.model.forward(img, data['fwd_poly'])
                pred_cps = output['pred_polys'][-1]
                pred_polys = self.spline.sample_point(pred_cps)
                pred_polys = pred_polys.data.cpu().numpy()
                # print(pred_polys.shape)
                # Get IoU
                iou = 0
                orig_poly = data['orig_poly']
                for i in range(pred_polys.shape[0]):
                    curr_pred_poly = utils.poly01_to_poly0g(pred_polys[i], self.model.grid_size)
                    curr_gt_poly = utils.poly01_to_poly0g(orig_poly[i], self.model.grid_size)
                    i, masks = metrics.iou_from_poly(np.array(curr_pred_poly, dtype=np.int32),
                                                           np.array(curr_gt_poly, dtype=np.int32),
                                                           self.model.grid_size,
                                                           self.model.grid_size)
                    iou += i
                iou = iou / pred_polys.shape[0]
                ious.append(iou)
                del (output)
                del (pred_polys)
            iou = np.mean(ious)
            self.val_writer.add_scalar('iou', float(iou), self.global_step)
            print '[VAL] IoU: %f' % iou
            masks = np.tile(masks, [1, 1, 1, 3])  # Make [2, H, W, 3]
            img = (data['img'].cpu().numpy()[-1, ...] * 255).astype(np.uint8)
            img = np.transpose(img, [1, 2, 0])  # Make [H, W, 3]
            #self.val_writer.add_image('pred_mask', masks[0], self.global_step)
            #self.val_writer.add_image('gt_mask', masks[1], self.global_step)
            #self.val_writer.add_image('image', img, self.global_step)
        self.model.train()


if __name__ == '__main__':
    print '==> Parsing Args'
    args = get_args()
    print '==> Init Trainer'
    trainer = Trainer(args)
    print '==> Start Loop over trainer'
    trainer.loop()

