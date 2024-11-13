import torch.nn as nn

import math


class ViTNeck(nn.Module):
    """

    A neck module for scaling ViT plain feature maps.

    """

    def __init__(self, in_channels, out_channels, scales=[4, 2, 1]):
        super().__init__()

        assert isinstance(in_channels, list)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channel in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(
                    in_channel,
                    out_channels,
                    kernel_size=1
                    ))
        for _ in range(self.num_outs):
            self.output_convs.append(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1
                    ))
            
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.uniform_(m.bias, 0)

    def forward(self, inputs, cls_token=True):
        assert len(inputs) == len(self.in_channels)

        inputs = list(inputs)
        
        batch_size = inputs[-1].shape[0]
        for i, feat in enumerate(inputs):
            if cls_token:
                feat = feat[:,1:,:]

            height = width = int(math.sqrt(feat.shape[1]))
            feat = feat.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            inputs[i] = feat

        inputs = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(self.num_outs)]

        outs = []
        for feat, out_scale, output_conv in zip(inputs, self.scales, self.output_convs):
            x_resize = nn.functional.interpolate(feat, scale_factor=out_scale, mode='bilinear')
            outs.append(output_conv(x_resize))
            
        return tuple(outs)


class DenseCLIPNeck(nn.Module):
    def __init__(self, width):
        super().__init__()

        self.fpn_dim = width
        self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(self.fpn_dim),
                nn.GELU(),
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)  
            
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.uniform_(m.bias, 0)

    def forward(self, inputs, cls_token=True):
        assert len(inputs) == 3

        inputs = list(inputs)
        
        batch_size = inputs[-1].shape[0]
        for i, feat in enumerate(inputs):
            if cls_token:
                feat = feat[:,1:,:]

            height = width = int(math.sqrt(feat.shape[1]))
            feat = feat.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            inputs[i] = feat

        ops = [self.fpn1, self.fpn2, self.fpn3]
        for i in range(len(inputs)):
            inputs[i] = ops[i](inputs[i])
            
        return tuple(inputs)