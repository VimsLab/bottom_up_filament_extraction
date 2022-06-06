from .resnet import resnet50, resnet101
import torch.nn as nn
import torch
from .globalNet import globalNet
from .refineNet import refineNet
from .segNet import segNet
from .multiNet import multiNet
from .unet import UNet
import sys

__all__ = ['CPN50', 'CPN101']

class CPN(nn.Module):
    def __init__(self, resnet, output_shape, num_class, inter=False, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        channel_settings_seg_net = [2048, 1024, 512, 256, 64]
        # self.resnet = resnet
        # self.globalNet = globalNet(channel_settings, output_shape, 2)
        # self.segNet = segNet(channel_settings_seg_net,1)
        # self.interSegNet = segNet(channel_settings_seg_net,2)
        # self.directionNet = segNet(channel_settings_seg_net,6)
        self.embedNet_inter = UNet(2,1,2)
        # self.embedNet_inter = UNet(3,1,2)
        # self.offsetNet = globalNet(channel_settings, output_shape, 6)
        # self.multiNet = multiNet(channel_settings[3], output_shape)
        # self.unet = UNet(2, 2,2)
        self.unet = UNet(2, 2,2)
        # self.internet = UNet(2, 1,2)
        # self.unet = UNet(3, 2,6)
        # self.conv = nn.Conv2d(16, 3, kernel_size=3, stride=1,
        #              padding=1, bias=True)
        self.inter = inter
        # self.refineNet = refineNet(channel_settings[-1], output_shape, 3)

        # self.global_net = globalNet(channel_settings, output_shape, num_class)
        # self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)


    def forward(self, x):
        # res_out = self.resnet(x)

        # import pdb; pdb.set_trace()
        # feature_foward, global_fms, intermediate_global_outs, _ = self.globalNet(res_out)

        # _, _, global_outs, offsets = self.offsetNet(res_out)

        # control_short_offset_pred = offsets[3][:,0:2,:,:]
        # next_offset_pred = offsets[3][:,2:4,:,:]
        # end_short_offset_pred = offsets[3][:,4:6,:,:]
        # if not self.inter:
        #     # _, _, mask_pred = self.segNet(res_out)
        #     # mask_pred = torch.sigmoid(mask_pred)
        #     mask_pred, directionNet_predict = self.unet(x)
        # else:
        #     mask_pred, directionNet_predict = self.unet(x)
        mask_pred, interseg_out = self.unet(x)

        embed_predict, _ = self.embedNet_inter(x)


        # _, _, interseg_out= self.interSegNet(res_out)
        
        interseg_out = torch.sigmoid(interseg_out)
        # directionNet_predict = torch.sigmoid(directionNet_predict)
        mask_pred = torch.sigmoid(mask_pred)
        # _, _, embed_predict= self.embedNet(res_out)
        # embed_predict = torch.sigmoid(embed_predict)
        # embed_predict = self.conv(embed_predict)
        # embed_predict = torch.sigmoid(embed_predict)

        # embed_predict = torch.nn.functional.normalize(embed_predict, p=2, dim=-1)
        
        ###################################################################
        # embed_predict_permuted = embed_predict.permute(0,2,3,1)
        # embed_predict_tmp = torch.flatten(embed_predict_permuted, end_dim=2)
        # embed_predict_tmp = torch.nn.functional.normalize(embed_predict_tmp, p=2, dim=1)
        # normed_out_tag = torch.nn.functional.normalize(embed_predict, p=2, dim=2)
        # embed_predict_tmp = embed_predict_tmp.view(embed_predict_permuted.shape)
        # embed_predict = embed_predict_tmp.permute(0,3,1,2)
        ###########################################################################
        #
        # _, _, directionNet_predict= self.directionNet(res_out)
        # directionNet_predict = torch.sigmoid(directionNet_predict)

        # directionNet_predict = directionNet_predict

        # _, control_point_pred, end_point_pred, long_offset_pred, \
        #     next_offset_pred_, prev_offset_pred, control_short_offset_pred_, end_short_offset_pred_, \
        #     end_point_embeding, control_point_embeding = self.multiNet(feature_foward)

        # refine_out = self.refineNet(global_fms)
        

        
        # mask_pred = mask_pred
        # embed_predict = embed_predict
        # return mask_pred, control_point_pred, end_point_pred, long_offset_pred,\
        #        next_offset_pred, prev_offset_pred, control_short_offset_pred, end_short_offset_pred,\
        #        end_point_embeding, control_point_embeding, global_outs, embed_predict, directionNet_predict, interseg_out
        return mask_pred, embed_predict, interseg_out, interseg_out
        # return mask_pred, embed_predict

class CPN_2(nn.Module):
    def __init__(self, resnet, output_shape, num_class, inter=False, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        channel_settings_seg_net = [2048, 1024, 512, 256, 64]
        self.resnet = resnet
        # self.globalNet = globalNet(channel_settings, output_shape, 2)
        self.segNet = segNet(channel_settings_seg_net,1)
        self.interSegNet = segNet(channel_settings_seg_net,2)
        self.directionNet = segNet(channel_settings_seg_net,6)
        self.embedNet = segNet(channel_settings_seg_net, 1)
        # self.offsetNet = globalNet(channel_settings, output_shape, 6)
        # self.multiNet = multiNet(channel_settings[3], output_shape)
        self.unet = UNet(3, 2,16)
        self.conv = nn.Conv2d(16, 3, kernel_size=3, stride=1,
                     padding=1, bias=True)
        self.inter = inter
        # self.refineNet = refineNet(channel_settings[-1], output_shape, 3)

        # self.global_net = globalNet(channel_settings, output_shape, num_class)
        # self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)


    def forward(self, x):
        res_out = self.resnet(x)

        # import pdb; pdb.set_trace()
        # feature_foward, global_fms, intermediate_global_outs, _ = self.globalNet(res_out)

        # _, _, global_outs, offsets = self.offsetNet(res_out)

        # control_short_offset_pred = offsets[3][:,0:2,:,:]
        # next_offset_pred = offsets[3][:,2:4,:,:]
        # end_short_offset_pred = offsets[3][:,4:6,:,:]
        if not self.inter:
            # _, _, mask_pred = self.segNet(res_out)
            # mask_pred = torch.sigmoid(mask_pred)
            mask_pred, _ = self.unet(x)
        else:
            mask_pred, _ = self.unet(x)

        _, _, interseg_out= self.interSegNet(res_out)
        
        interseg_out = torch.sigmoid(interseg_out)
        _, _, embed_predict= self.embedNet(res_out)
        # embed_predict = torch.sigmoid(embed_predict)
        # embed_predict = self.conv(embed_predict)
        # embed_predict = torch.sigmoid(embed_predict)

        # embed_predict = torch.nn.functional.normalize(embed_predict, p=2, dim=-1)
        
        ###################################################################
        # embed_predict_permuted = embed_predict.permute(0,2,3,1)
        # embed_predict_tmp = torch.flatten(embed_predict_permuted, end_dim=2)
        # embed_predict_tmp = torch.nn.functional.normalize(embed_predict_tmp, p=2, dim=1)
        # normed_out_tag = torch.nn.functional.normalize(embed_predict, p=2, dim=2)
        # embed_predict_tmp = embed_predict_tmp.view(embed_predict_permuted.shape)
        # embed_predict = embed_predict_tmp.permute(0,3,1,2)
        ###########################################################################
        #
        _, _, directionNet_predict= self.directionNet(res_out)
        directionNet_predict = torch.sigmoid(directionNet_predict)

        directionNet_predict = directionNet_predict

        # _, control_point_pred, end_point_pred, long_offset_pred, \
        #     next_offset_pred_, prev_offset_pred, control_short_offset_pred_, end_short_offset_pred_, \
        #     end_point_embeding, control_point_embeding = self.multiNet(feature_foward)

        # refine_out = self.refineNet(global_fms)
        

        
        # mask_pred = mask_pred
        # embed_predict = embed_predict
        # return mask_pred, control_point_pred, end_point_pred, long_offset_pred,\
        #        next_offset_pred, prev_offset_pred, control_short_offset_pred, end_short_offset_pred,\
        #        end_point_embeding, control_point_embeding, global_outs, embed_predict, directionNet_predict, interseg_out
        return mask_pred, embed_predict, directionNet_predict, interseg_out
        # return mask_pred, embed_predict

def CPN50(out_size,num_class, inter =False, pretrained=True):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, output_shape=out_size,num_class=num_class, inter=inter, pretrained=pretrained)
    return model

def CPN101(out_size,num_class,pretrained=True):
    res101 = resnet101(pretrained=pretrained)
    model = CPN(res101, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model


# class CPN(nn.Module):
#     def __init__(self, resnet, output_shape, num_class, pretrained=True):
#         super(CPN, self).__init__()
#         channel_settings = [2048, 1024, 512, 256]
#         self.resnet = resnet
#         self.seg_net = segNet(channel_settings, output_shape, num_class = 1)
#         self.global_net = globalNet(channel_settings, output_shape, num_class)
#         self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)


#     def forward(self, x):
#         res_out = self.resnet(x)
#         global_fms, global_outs, global_offset = self.global_net(res_out)
#         refine_out = self.refine_net(global_fms)
#         seg_out = self.seg_net(res_out)

#         # import pdb; pdb.set_trace()
#         return global_outs,refine_out, global_offset, seg_out

# def CPN50(out_size,num_class,pretrained=True):
#     res50 = resnet50(pretrained=pretrained)
#     model = CPN(res50, output_shape=out_size,num_class=num_class, pretrained=pretrained)
#     return model

# def CPN101(out_size,num_class,pretrained=True):
#     res101 = resnet101(pretrained=pretrained)
#     model = CPN(res101, output_shape=out_size,num_class=num_class, pretrained=pretrained)
#     return model
