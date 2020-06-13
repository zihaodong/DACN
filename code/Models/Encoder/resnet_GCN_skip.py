# from resnet import ResNet, Bottleneck
from resnet_101 import ResNet,Bottleneck
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
import Utils.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #self.sigmoid = nn.Sigmoid() ##add dongzihao
    def forward(self, x):
        x = self.conv1(x)
        res  = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)

class CAB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # high, low
        x = torch.cat([x1,x2],dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res


class SkipResnet50(nn.Module):
    def __init__(self, concat_channels=64,
                 final_dim=128,
                 nInputChannels=3,
                 cnn_feature_grids=None,
                 classifier=""):

        super(SkipResnet50, self).__init__()

        # Default transform for all torchvision models
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.concat_channels = concat_channels
        self.final_dim = final_dim
        self.feat_size = 28
        self.cnn_feature_grids = cnn_feature_grids
        self.image_feature_dim = 256
        self.nInputChannels = nInputChannels
        self.classifier =classifier
        # self.resnet = ResNet(Bottleneck, layers=[3, 4, 6, 3], strides=[1, 2, 1, 1],
        #                      nInputChannels=nInputChannels,
        #                      dilations=[1, 1, 2, 4],
        #                      classifier=self.classifier)
        self.resnet_features = ResNet(Bottleneck, [3, 4, 23, 3])


        concat1 = nn.Conv2d(64, concat_channels, kernel_size=3, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(concat_channels)
        relu1 = nn.ReLU(inplace=True)

        self.conv1_concat = nn.Sequential(concat1, bn1, relu1)

        concat2 = nn.Conv2d(256, concat_channels, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(concat_channels)
        relu2 = nn.ReLU(inplace=True)
        up2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        self.res1_concat = nn.Sequential(concat2, bn2, relu2)
        self.res1_concat_up = nn.Sequential(concat2, bn2, relu2, up2)

        concat3 = nn.Conv2d(512, concat_channels, kernel_size=3, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(concat_channels)
        relu3 = nn.ReLU(inplace=True)
        up3 = torch.nn.Upsample(scale_factor=4, mode='bilinear')

        self.res2_concat = nn.Sequential(concat3, bn3, relu3)

        self.res2_concat_up = nn.Sequential(concat3, bn3, relu3, up3)

        concat4 = nn.Conv2d(2048, concat_channels, kernel_size=3, padding=1, bias=False)
        bn4 = nn.BatchNorm2d(concat_channels)
        relu4 = nn.ReLU(inplace=True)
        up4 = torch.nn.Upsample(scale_factor=4, mode='bilinear')

        self.res4_concat = nn.Sequential(concat4, bn4, relu4)
        self.res4_concat_up = nn.Sequential(concat4, bn4, relu4, up4)

        # Different from original, original used maxpool
        # Original used no activation here
        conv_final_1 = nn.Conv2d(4*concat_channels, 128, kernel_size=3, padding=1, stride=2,
            bias=False)
        bn_final_1 = nn.BatchNorm2d(128)
        conv_final_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False)
        bn_final_2 = nn.BatchNorm2d(128)
        conv_final_3 = nn.Conv2d(128, final_dim, kernel_size=3, padding=1, bias=False)
        bn_final_3 = nn.BatchNorm2d(final_dim)

        self.conv_final = nn.Sequential(conv_final_1, bn_final_1, conv_final_2, bn_final_2,
            conv_final_3, bn_final_3)

        self.final_dim = 64 * 4

        self.num_class = 256

        # self.rrb_d_1 = RRB(64, self.final_dim)
        # self.rrb_d_2 = RRB(64, self.final_dim)
        # self.rrb_d_3 = RRB(64, self.final_dim)
        # self.rrb_d_4 = RRB(64, self.final_dim)

        # ## this is for boarder net work
        # self.rrb_db_1 = RRB(64, self.final_dim)
        # self.rrb_db_2 = RRB(64, self.final_dim)
        # self.rrb_db_3 = RRB(64, self.final_dim)
        # self.rrb_db_4 = RRB(64, self.final_dim)

        # self.rrb_trans_1 = RRB(self.final_dim,self.final_dim)
        # self.rrb_trans_2 = RRB(self.final_dim,self.final_dim)
        # self.rrb_trans_3 = RRB(self.final_dim,self.final_dim)
        # self.upsample = nn.Upsample(scale_factor=2,mode="bilinear")
        # self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear")
        # self.upsample_8 = nn.Upsample(scale_factor=8, mode="bilinear")

        self.layer0 = nn.Sequential(self.resnet_features.conv1, self.resnet_features.bn1,
                                    self.resnet_features.relu #self.resnet_features.conv1,
                                    #self.resnet_features.bn1, self.resnet_features.relu
                                    )
        self.layer1 = nn.Sequential(self.resnet_features.maxpool, self.resnet_features.layer1)
        self.layer2 = self.resnet_features.layer2
        self.layer3 = self.resnet_features.layer3
        self.layer4 = self.resnet_features.layer4

        # this is for smooth network
        self.out_conv = nn.Conv2d(2048,self.num_class,kernel_size=1,stride=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cab1 = CAB(self.num_class*2,self.num_class)
        self.cab2 = CAB(self.num_class*2,self.num_class)
        self.cab3 = CAB(self.num_class*2,self.num_class)
        self.cab4 = CAB(self.num_class*2,self.num_class)

        self.rrb_d_1 = RRB(256, self.num_class)
        self.rrb_d_2 = RRB(512, self.num_class)
        self.rrb_d_3 = RRB(1024, self.num_class)
        self.rrb_d_4 = RRB(2048, self.num_class)

        self.upsample = nn.Upsample(scale_factor=2,mode="bilinear")
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.upsample_8 = nn.Upsample(scale_factor=8, mode="bilinear")

        self.rrb_u_1 = RRB(self.num_class,self.num_class)
        self.rrb_u_2 = RRB(self.num_class,self.num_class)
        self.rrb_u_3 = RRB(self.num_class,self.num_class)
        self.rrb_u_4 = RRB(self.num_class,self.num_class)


        ## this is for boarder net work
        self.rrb_db_1 = RRB(256, self.num_class)
        self.rrb_db_2 = RRB(512, self.num_class)
        self.rrb_db_3 = RRB(1024, self.num_class)
        self.rrb_db_4 = RRB(2048, self.num_class)

        self.rrb_trans_1 = RRB(self.num_class,self.num_class)
        self.rrb_trans_2 = RRB(self.num_class,self.num_class)
        self.rrb_trans_3 = RRB(self.num_class,self.num_class)


        self.edge_annotation_concat_channels = 64 * 4

        #128+2
        edge_annotation_cnn_tunner_1 = nn.Conv2d(128+1, self.edge_annotation_concat_channels, kernel_size=3, padding=1, bias=False)
        edge_annotation_cnn_tunner_bn_1 = nn.BatchNorm2d(self.edge_annotation_concat_channels)
        edge_annotation_cnn_tunner_relu_1 = nn.ReLU(inplace=True)

        edge_annotation_cnn_tunner_2 = nn.Conv2d(self.edge_annotation_concat_channels, self.edge_annotation_concat_channels, kernel_size=3,
                                                 padding=1, bias=False)
        edge_annotation_cnn_tunner_bn_2 = nn.BatchNorm2d(self.edge_annotation_concat_channels)
        edge_annotation_cnn_tunner_relu_2 = nn.ReLU(inplace=True)

        self.edge_annotation_concat = nn.Sequential(edge_annotation_cnn_tunner_1,
                                                    edge_annotation_cnn_tunner_bn_1,
                                                    edge_annotation_cnn_tunner_relu_1,
                                                    edge_annotation_cnn_tunner_2,
                                                    edge_annotation_cnn_tunner_bn_2,
                                                    edge_annotation_cnn_tunner_relu_2)

    def reload(self, path):

        if self.nInputChannels != 3:
            print "Reloading resnet for: ", path , ", InputChannel: ", self.nInputChannels
            # model_full = ResNet(Bottleneck, layers=[3, 4, 6, 3], strides=[1, 2, 1, 1],
            #                  nInputChannels=3,
            #                  dilations=[1, 1, 2, 4]).to(device)
            model_full = ResNet(Bottleneck, [3, 4, 23, 3])
            model_full.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
            self.resnet_features.load_pretrained_ms(model_full, nInputChannels=self.nInputChannels)
            del(model_full)
        else:
            print "Reloading resnet from: ", path
            self.resnet_features.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage),strict=False)


    def edge_annotation_cnn(self, feature):

        final_feature_map = self.edge_annotation_concat(feature)

        return final_feature_map.permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids[-1]**2, self.edge_annotation_concat_channels)

    def forward(self, x, return_final_res_feature=True):
        x = self.normalize(x)
        #print(x.shape)
        # Normalization

        # conv1_f, layer1_f, layer2_f, layer3_f, layer4_f , layer5_f = self.resnet(x)

        # # torch.Size([1, 64, 112, 112])
        # conv1_f_gcn = self.conv1_concat(conv1_f) #.permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids[0]**2, 64)
        # # torch.Size([1, 64, 56, 56])
        # layer1_f_gcn = self.res1_concat(layer1_f)#.permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids[1]**2, 64)
        # # torch.Size([1, 64, 28, 28])
        # layer2_f_gcn = self.res2_concat(layer2_f)#.permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids[2]**2, 64)
        # # torch.Size([1, 64, 28, 28])
        # layer4_f_gcn = self.res4_concat(layer4_f)#.permute(0, 2, 3, 1).view(-1, self.cnn_feature_grids[3]**2, 64)

        # res1 = self.rrb_db_1(conv1_f_gcn)
        # res1 = self.rrb_trans_1(res1 + self.upsample(self.rrb_db_2(layer1_f_gcn)))
        # res1 = self.rrb_trans_2(res1 + self.upsample_4(self.rrb_db_3(layer2_f_gcn)))
        # res1 = self.rrb_trans_3(res1 + self.upsample_4(self.rrb_db_4(layer4_f_gcn)))
        # final_features = self.conv_final(res1)
        # return conv1_f_gcn, layer1_f_gcn, layer2_f_gcn, layer4_f_gcn, layer5_f, final_features
        #print('x:',x.shape)
        f0 = self.layer0(x)  # 256  112
        #print('f0:',f0.shape)
        f1 = self.layer1(f0)  # 128 56
        f2 = self.layer2(f1)  # 64  28
        #print('f2:',f2.shape)
        f3 = self.layer3(f2)  # 32  28
        #print('f3:',f3.shape)
        f4 = self.layer4(f3)  # 16  28
        #print('f4:',f4.shape)
        # for border network
        res1 = self.rrb_db_1(f1)
        #print('f1:',f1.shape)
        res1 = self.rrb_trans_1(res1 + self.upsample(self.rrb_db_2(f2)))
        res1 = self.rrb_trans_2(res1 + self.upsample(self.rrb_db_3(f3)))
        res1 = self.rrb_trans_3(res1 + self.upsample(self.rrb_db_4(f4)))
        res1 = self.conv_final(res1)
        # print (res1.size())
        # for smooth network
        res2 = self.out_conv(f4)
        res2 = self.global_pool(res2)  #
        res2 = nn.Upsample(size=f4.size()[2:],mode="nearest")(res2)

        f4 = self.rrb_d_4(f4)
        res2 = self.cab4([res2,f4])
        #res2 = f4
        res2 = self.rrb_u_1(res2)

        f3 = self.rrb_d_3(f3)
        res2 = self.cab3([res2,f3])
        #res2 = f3
        res2 =self.rrb_u_2(res2)

        f2 = self.rrb_d_2(f2)
        res2 = self.cab2([res2, f2])
        #res2 = f2
        res2 =self.rrb_u_3(res2)

        f1 = self.rrb_d_1(f1)
        res2 = self.cab1([self.upsample(res2), f1])
        #res2 = f1
        res2 = self.rrb_u_4(res2)
        res2 = self.conv_final(res2)


        return res1, res2


    def sampling(self, ids, features):

        cnn_out_feature = []
        for i in range(ids.size()[1]):
            id =  ids[:, i, :]

            cnn_out = utils.gather_feature(id, features[i])
            cnn_out_feature.append(cnn_out)

        concat_features = torch.cat(cnn_out_feature, dim=2)

        return concat_features


    def normalize(self, x):
        individual = torch.unbind(x, dim=0)
        out = []
        for x in individual:
            out.append(self.normalizer(x))

        return torch.stack(out, dim=0)

if __name__ == '__main__':
    model = SkipResnet50()
    model(torch.randn(1, 3, 224, 224))
