import torch
import json
import os
import argparse
import numpy as np
import warnings
import skimage.io as sio
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import sys
sys.path.append('.')
from Utils import utils
from active_spline import ActiveSplineTorch
from DataProvider import cityscape_from_eclipse_active_spline
from Models.GNN import poly_gnn
import timeit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print ('==> Using Devices %s' % (device))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--reload', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    return args


def get_data_loaders(opts, DataProvider):
    print 'Building dataloaders'

    dataset_val = DataProvider(split='val', opts=opts['train_val'], mode='oracle_test')

    val_loader = DataLoader(dataset_val, batch_size=opts['train_val']['batch_size'],
                            shuffle=False, num_workers=opts['train_val']['num_workers'],
                            collate_fn=cityscape_from_eclipse_active_spline.collate_fn)
    return val_loader


def override_options(opts):
    opts['mode'] = 'test'
    opts['temperature'] = 0.0
    opts['dataset']['train_val']['skip_multicomponent'] = False
    opts.pop('encoder_reload', None)
    opts['dataset']['train']['ext_points'] = opts['ext_points']
    opts['dataset']['train_val']['ext_points'] = opts['ext_points']
    opts['dataset']['train']['p_num'] = opts['p_num']
    opts['dataset']['train_val']['p_num'] = opts['p_num']
    opts['dataset']['train']['cp_num'] = opts['cp_num']
    opts['dataset']['train_val']['cp_num'] = opts['cp_num']
    opts['dataset']['train']['ext_points_pert'] = opts['ext_points_pert']
    opts['dataset']['train_val']['ext_points_pert'] = opts['ext_points_pert']
    return opts


class Tester(object):
    def __init__(self, args):
        self.opts = json.load(open(args.exp, 'r'))
        self.output_dir = args.output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join(self.opts['exp_dir'], 'preds')
        print '==> Clean output folder'
        if os.path.exists(self.output_dir): shutil.rmtree(self.output_dir)
        utils.create_folder(self.output_dir)
        self.opts = override_options(self.opts)
        self.val_loader = get_data_loaders(self.opts['dataset'], cityscape_from_eclipse_active_spline.DataProvider)
        self.spline = ActiveSplineTorch(self.opts['cp_num'], self.opts[u'p_num'], device=device)


        self.model = poly_gnn.PolyGNN(state_dim=self.opts['state_dim'],
                                      n_adj=self.opts['n_adj'],
                                      cnn_feature_grids=self.opts['cnn_feature_grids'],
                                      coarse_to_fine_steps=self.opts['coarse_to_fine_steps'],
                                      get_point_annotation=self.opts['get_point_annotation']
                                      ).to(device)

        print '==> Reloading Models'
        self.model.reload(args.reload, strict=False)

    def process_outputs(self, data, output, save=True):
        """
        Process outputs to get final outputs for the whole image
        Optionally saves the outputs to a folder for evaluation
        """

        #pred_spline = output['pred_polys']
        pred_spline = output['pred_polys']
        preds = self.spline.sample_point(pred_spline)
        torch.cuda.synchronize()
        preds = preds.cpu().numpy()

        pred_spline = pred_spline.cpu()
        pred_spline = pred_spline.numpy()

        instances = data['instance']
        polys = []
        for i, instance in enumerate(instances):
            poly = preds[i]
            poly = poly * data['patch_w'][i]
            poly[:, 0] += data['starting_point'][i][0]
            poly[:, 1] += data['starting_point'][i][1]


            pred_sp = pred_spline[i]
            pred_sp = pred_sp * data['patch_w'][i]
            pred_sp[:, 0] += data['starting_point'][i][0]
            pred_sp[:, 1] += data['starting_point'][i][1]

            instance['spline_pos'] = pred_sp.tolist()

            polys.append(poly)

            if save:
                img_h, img_w = instance['img_height'], instance['img_width']
                predicted_poly = []

                pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                utils.draw_poly(pred_mask, poly.astype(np.int))
                #utils.get_edge_mask(poly.astype(np.int),pred_mask)
                #utils.get_vertices_mask(poly.astype(np.int),pred_mask)
                predicted_poly.append(poly.tolist())

                gt_mask = utils.get_full_mask_from_instance(
                    self.opts['dataset']['train_val']['min_area'],
                    instance)

                instance['my_predicted_poly'] = predicted_poly
                instance_id = instance['instance_id']
                image_id = instance['image_id']

                pred_mask_fname = os.path.join(self.output_dir, '{}_pred.png'.format(instance_id))
                instance['pred_mask_fname'] = os.path.relpath(pred_mask_fname, self.output_dir)

                gt_mask_fname = os.path.join(self.output_dir, '{}_gt.png'.format(instance_id))
                instance['gt_mask_fname'] = os.path.relpath(gt_mask_fname, self.output_dir)

                instance['n_corrections'] = 0

                info_fname = os.path.join(self.output_dir, '{}_info.json'.format(instance_id))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sio.imsave(pred_mask_fname, pred_mask)
                    sio.imsave(gt_mask_fname, gt_mask)

                # print '==> dumping json'
                with open(info_fname, 'w') as f:
                    json.dump(instance, f, indent=2)

        return polys

    def test(self):
        print 'Starting testing'
        self.model.eval()

        # Leave LSTM in train mode
        times = []
        count = 0
        with torch.no_grad():
            for step, data in enumerate(tqdm(self.val_loader)):
                # Forward pass

                if self.opts['get_point_annotation']:
                    img = data['img'].to(device)
                    annotation = data['annotation_prior'].to(device).unsqueeze(1)
                    img = torch.cat([img, annotation], 1)
                else:
                    img = data['img'].to(device)

                start = timeit.default_timer()
                output = self.model(img,
                                    data['fwd_poly'])
                stop = timeit.default_timer()
                if count>0:
                    times.append(stop - start)

                if self.opts['coarse_to_fine_steps'] > 0:

                    output['pred_polys'] = output['pred_polys'][-1]
                # Bring everything to cpu/numpy
                for k in output.keys():
                    if k == 'pred_polys': continue
                    if k == 'edge_logits': continue
                    if k == 'vertex_logits': continue
                    output[k] = output[k].cpu().numpy()

                self.process_outputs(data, output, save=True)
                del (output)
                if count > 0:
                    print sum(times)/float(len(times))
                count = count + 1



if __name__ == '__main__':
    args = get_args()
    tester = Tester(args)
    tester.test()
