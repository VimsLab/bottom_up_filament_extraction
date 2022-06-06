import torch.nn as nn
import torch
import math

class multiNet(nn.Module):
    def __init__(self, channel_settings, output_shape):
        super(multiNet, self).__init__()
        self.channel_settings = channel_settings
        self.lateral = self._lateral(channel_settings)
        self.predict_segmask = self._predict(output_shape, num_class = 1)
        self.predict_endpoint = self._predict(output_shape, num_class = 1)
        self.predict_controlpoint = self._predict(output_shape, num_class = 1)
        self.predict_control_embedding = self._predict(output_shape, num_class = 1)
        self.predict_end_embedding = self._predict(output_shape, num_class = 1)
        self.predict_short_offset_control = self._predict(output_shape, num_class = 2)
        self.predict_short_offset_end = self._predict(output_shape, num_class = 2)
        self.predict_prev_offset = self._predict(output_shape, num_class = 2)
        self.predict_next_offset = self._predict(output_shape, num_class = 2)
        self.predict_long_offset = self._predict(output_shape, num_class = 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))

        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)

    def forward(self, x):

        feature = self.lateral(x)

        mask_pred =torch.sigmoid(self.predict_segmask(feature))
        control_point_pred =torch.sigmoid(self.predict_controlpoint(feature))
        end_point_pred =torch.sigmoid(self.predict_endpoint(feature))
        long_offset_pred =self.predict_long_offset(feature)
        next_offset_pred =self.predict_next_offset(feature)
        prev_offset_pred =self.predict_prev_offset(feature)
        control_short_offset_pred =self.predict_short_offset_control(feature)
        end_short_offset_pred =self.predict_short_offset_end(feature)


        control_point_embeding = self.predict_control_embedding(feature)
        end_point_embeding = self.predict_end_embedding(feature)



        return mask_pred, control_point_pred, end_point_pred, long_offset_pred, \
               next_offset_pred, prev_offset_pred, control_short_offset_pred, end_short_offset_pred, \
               end_point_embeding, control_point_embeding
