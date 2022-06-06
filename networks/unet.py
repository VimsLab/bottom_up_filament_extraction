import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   
""" Parts of the U-Net model """

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_classes2, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.up12 = Up(1024, 512 // factor, bilinear)
        self.up22 = Up(512, 256 // factor, bilinear)
        self.up32 = Up(256, 128 // factor, bilinear)
        self.up42 = Up(128, 64, bilinear)
        self.outc2 = OutConv(64, n_classes2)

        # self.outc3 = OutConv(n_classes2, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # out = torch.sigmoid(logits)

        x22 = self.up12(x5, x4)
        x22 = self.up22(x22, x3)
        x22 = self.up32(x22, x2)
        x22 = self.up42(x22, x1)
        logits2 = self.outc2(x22)
        # logits2 = self.outc3(logits2)
        # logits2 = torch.nn.functional.normalize(logits2, p=2, dim=1)
        # logits2 = torch.sigmoid(logits2)
        
        return logits, logits2

# class UNet(nn.Module):

#     def __init__(self, n_class, n_class_2):
#         # super().__init__()
#         super(UNet, self).__init__()
#         self.dconv_down1 = double_conv(3, 64)
#         self.dconv_down2 = double_conv(64, 128)
#         self.dconv_down3 = double_conv(128, 256)
#         self.dconv_down4 = double_conv(256, 512)        

#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
#         self.dconv_up3 = double_conv(256 + 512, 256)
#         self.dconv_up2 = double_conv(128 + 256, 128)
#         self.dconv_up1 = double_conv(128 + 64, 64)

#         self.dconv_up3_2 = double_conv(256 + 512, 256)
#         self.dconv_up2_2 = double_conv(128 + 256, 128)
#         self.dconv_up1_2 = double_conv(128 + 64, 64)
        
#         self.conv_last = nn.Conv2d(64, n_class, 1)
#         self.conv_last_2 = nn.Conv2d(64, n_class_2, 1)
        
        
#     def forward(self, x):
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool(conv1)

#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)
        
#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)   
        
#         x = self.dconv_down4(x)
        
#         x = self.upsample(x)        
#         x = torch.cat([x, conv3], dim=1)


        
#         x_1 = self.dconv_up3(x)
#         x_1 = self.upsample(x_1)        
#         x_1 = torch.cat([x_1, conv2], dim=1) 

#         x_2 = self.dconv_up3_2(x)
#         x_2 = self.upsample(x_2)        
#         x_2 = torch.cat([x_2, conv2], dim=1)       

#         x_1 = self.dconv_up2(x_1)
#         x_1 = self.upsample(x_1)        
#         x_1 = torch.cat([x_1, conv1], dim=1)   

#         x_2 = self.dconv_up2_2(x_2)
#         x_2 = self.upsample(x_2)        
#         x_2 = torch.cat([x_2, conv1], dim=1)   
        
#         x_1 = self.dconv_up1(x_1)

#         x_2 = self.dconv_up1_2(x_2)
#         # print("--------conv1---------")
#         # print(torch.sum(torch.isnan(x_2)))
#         # print("--------conv1---------")
#         if torch.sum(torch.isnan(x_1))>0:
#             import pdb; pdb.set_trace()
#         if torch.sum(torch.isnan(x_2))>0:
#             import pdb; pdb.set_trace()
#         out = self.conv_last(x_1)
#         out_2 = self.conv_last_2(x_2)

#         out = torch.sigmoid(out)
#         # out_2 = torch.nn.functional.normalize(out_2, p=2, dim=1)
#         return out, out_2