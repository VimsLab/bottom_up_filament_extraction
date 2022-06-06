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
    def __init__(self, resnet, output_shape, num_class, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        channel_settings_seg_net = [2048, 1024, 512, 256, 64]
        self.resnet = resnet
        self.globalNet = globalNet(channel_settings, output_shape, 2)
        self.segNet = segNet(channel_settings_seg_net,1)
        self.directionNet = segNet(channel_settings_seg_net,6)
        self.embedNet = segNet(channel_settings_seg_net, 9)
        self.offsetNet = globalNet(channel_settings, output_shape, 6)
        self.multiNet = multiNet(channel_settings[3], output_shape)
        # self.refineNet = refineNet(channel_settings[-1], output_shape, 3)

        # self.global_net = globalNet(channel_settings, output_shape, num_class)
        # self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)


    def forward(self, x):
        res_out = self.resnet(x)

        # import pdb; pdb.set_trace()
        feature_foward, global_fms, intermediate_global_outs, _ = self.globalNet(res_out)

        _, _, global_outs, offsets = self.offsetNet(res_out)

        control_short_offset_pred = offsets[3][:,0:2,:,:]
        next_offset_pred = offsets[3][:,2:4,:,:]
        end_short_offset_pred = offsets[3][:,4:6,:,:]

        _, _, mask_pred= self.segNet(res_out)
        
        _, _, embed_predict= self.embedNet(res_out)
        _, _, directionNet_predict= self.directionNet(res_out)
        mask_pred = mask_pred
        embed_predict = embed_predict
        directionNet_predict = directionNet_predict

        _, control_point_pred, end_point_pred, long_offset_pred, \
            next_offset_pred_, prev_offset_pred, control_short_offset_pred_, end_short_offset_pred_, \
            end_point_embeding, control_point_embeding = self.multiNet(feature_foward)
        # refine_out = self.refineNet(global_fms)

        return mask_pred, control_point_pred, end_point_pred, long_offset_pred,\
               next_offset_pred, prev_offset_pred, control_short_offset_pred, end_short_offset_pred,\
               end_point_embeding, control_point_embeding, global_outs, embed_predict, directionNet_predict

def CPN50(out_size,num_class,pretrained=True):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, output_shape=out_size,num_class=num_class, pretrained=pretrained)
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
