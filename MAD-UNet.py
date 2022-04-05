import torch.nn as nn
import torch
import os


class SCSELayer(nn.Module):
    def __init__(self, channel):
        super(SCSELayer, self).__init__()
        self.cse_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.cse_fc = nn.Sequential(
            nn.Linear(channel, channel // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 2, channel, bias=False),
            nn.Sigmoid()
        )
        self.sse_all = nn.Sequential(
            nn.Conv3d(channel, 1, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, z, w, h = x.size()
        cse_y = self.cse_avg_pool(x).view(b, c)
        cse_y = self.cse_fc(cse_y).view(b, c, 1, 1, 1)
        sse_y = self.sse_all(x)
        return x * cse_y.expand_as(x) + x * sse_y.expand_as(x)


class Modified3DUNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=16):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.3)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsacle1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsacle2 = nn.Upsample(scale_factor=8, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.output4 = nn.Sequential(  # decoder2的输出
            nn.Conv3d(16, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )

        # SE
        self.se1 = SCSELayer(16)
        self.se2 = SCSELayer(32)
        self.se3 = SCSELayer(64)
        self.se4 = SCSELayer(128)

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)

        self.ds1_1x1_conv3d = nn.Conv3d(self.base_n_filter * 16, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        # print(out.shape)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)
        se1 = self.se1(out)

        # Level 2 context pathway
        out = self.conv3d_c2(se1)
        # print(out.shape)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2

        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        se2 = self.se2(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(se2)
        # print(out.shape)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3

        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        se3 = self.se3(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(se3)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        se4 = self.se4(out)
        # print(out.shape)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(se4)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        # print(out.shape)

        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        # print(out.shape)
        # out = self.conv3d_l0(out)
        # print(out.shape)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)

        resup1 = self.conv3d_l1(out)

        out = self.conv_norm_lrelu_l1(out)
        # ds1 = out
        out = self.conv3d_l1(out)
        out = resup1 + out
        # print(out.shape)
        se5 = self.se4(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(se5)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        resup2 = self.conv3d_l2(out)

        out = self.conv_norm_lrelu_l2(out)
        # ds2 = out
        out = self.conv3d_l2(out)
        out = resup2 + out
        se6 = self.se3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(se6)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        resup3 = self.conv3d_l3(out)
        out = self.conv_norm_lrelu_l3(out)
        # ds3 = out
        out = self.conv3d_l3(out)
        # print(out.shape)
        out = resup3 + out
        se7 = self.se2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(se7)
        # print(out.shape)
        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        # print(out.shape)
        resup4 = self.conv3d_l4(out)
        # print(resup4.shape)
        out = self.conv_norm_lrelu_l4(out)
        # print(out.shape)

        # print(out.shape)
        out = self.conv3d_l4(out)
        out_pred = resup4 + out

        out = self.output4(out_pred)
        return out


import torch.nn as nn
import torch
import os


class Modified3DUNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=16):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.3)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsacle1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsacle2 = nn.Upsample(scale_factor=8, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.output4 = nn.Sequential(  # decoder2的输出
            nn.Conv3d(16, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )
        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)

        self.ds1_1x1_conv3d = nn.Conv3d(self.base_n_filter * 16, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)

        self.con64_16 = self.conv_norm_lrelu(64, 16)
        self.conv16_16 = self.conv_norm_lrelu(16, 16)

        self.attention1 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=1), nn.InstanceNorm3d(16), nn.LeakyReLU(),

            nn.Conv3d(16, 16, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.attention2 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=1), nn.InstanceNorm3d(16), nn.LeakyReLU(),

            nn.Conv3d(16, 16, kernel_size=3, padding=1), nn.Sigmoid()
        )

        self.attention3 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=1), nn.InstanceNorm3d(16), nn.LeakyReLU(),

            nn.Conv3d(16, 16, kernel_size=3, padding=1), nn.Sigmoid()
        )

        self.attention4 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=1), nn.InstanceNorm3d(16), nn.LeakyReLU(),

            nn.Conv3d(16, 16, kernel_size=3, padding=1), nn.Sigmoid()
        )

        self.re1 = nn.Sequential(


            nn.Conv3d(16, 16, kernel_size=3, padding=1), nn.InstanceNorm3d(16), nn.LeakyReLU(),
        )

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())



    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        # print(out.shape)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        # print(out.shape)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        # print(out.shape)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        # print(out.shape)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        # print(out.shape)

        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        # print(out.shape)
        # out = self.conv3d_l0(out)
        # print(out.shape)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        ds1 = out
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        # print(out.shape)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        # print(out.shape)
        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        # print(out.shape)
        out = self.conv_norm_lrelu_l4(out)
        # print(out.shape)
        out_pred = self.conv3d_l4(out)
        # print(out_pred.shape)
        ds1_1x1_conv =self.ds1_1x1_conv3d(ds1)
        ds1_up = self.upsacle2(ds1_1x1_conv)
        # print(ds1_up.shape)
        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds2_up = self.upsacle1(ds2_1x1_conv)
        # print(ds2_up.shape)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds3up = self.upsacle(ds3_1x1_conv)
        # print(ds3up.shape)
        # ds1_2_3 = ds3up + ds2_up+ds1_up
        # out = out_pred +ds1_2_3
        out = torch.cat([ds3up, ds1_up, ds2_up, out_pred],dim=1)
        out = self.con64_16(out)
        out = self.conv16_16(out)
        print(out.shape)
        att1 = self.attention1(torch.cat([out, ds1_up], dim=1))
        att2 = self.attention2(torch.cat([out, ds2_up], dim=1))
        att3 = self.attention3(torch.cat([out, ds3up], dim=1))
        att4 = self.attention2(torch.cat([out, out_pred], dim=1))
        att1 = att1 * out
        att2 = att2 * out
        att3 = att3 * out
        att4 = att4 * out
        att1 = att1 + ds1_up
        att2 = att2 + ds2_up
        att3 = att3 + ds3up
        att4 = att4 + out_pred
        a1 = self.re1(att1)
        a2 = self.re1(att2)
        a3 = self.re1(att3)
        a4 = self.re1(att4)
        output = torch.cat([a1, a2, a3, a4], dim=1)
        output1 = self.con64_16(output)
        print(a1.shape)
        print(a2.shape)
        print(a3.shape)
        print(a4.shape)

        out = self.output4(output1)
        return out



import torch.nn as nn
import torch
import os
import torch.nn.functional as F
# from att import Attention_block
class before_sp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(before_sp, self).__init__()
        self.bconv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(),
        )

    def forward(self, input):
        return self.bconv(input)
class object_dp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(object_dp, self).__init__()
        self.tconv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2),
            nn.PReLU(),
        )

    def forward(self, input):
        return self.tconv(input)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi

        return out

class Modified3DUNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=16):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.3)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsacle1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsacle2 = nn.Upsample(scale_factor=8, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.output4 = nn.Sequential(  # decoder2的输出
            nn.Conv3d(16, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )
        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        self.cc1 = before_sp(32, 16)
        self.uu1 = object_dp(32, 16)
        self.cc2 = before_sp(64, 32)
        self.uu2 = object_dp(64, 32)
        self.cc3 = before_sp(128, 64)
        self.uu3 = object_dp(128, 64)
        self.cc4 = before_sp(256, 128)
        self.uu4 = object_dp(256, 128)
        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)

        self.ds1_1x1_conv3d = nn.Conv3d(self.base_n_filter * 16, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)

        self.p1 = nn.Conv3d(256, 128, kernel_size=1, stride=1, padding=0,
                            bias=False)
        self.p3 = nn.Conv3d(64, 128, kernel_size=1, stride=1, padding=0,
                            bias=False)
        self.p4 = nn.Conv3d(16, 128, kernel_size=1, stride=1, padding=0,
                            bias=False)

        self.fuse1 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.att1 = Attention_block(64, 128, 64)
        self.att2 = Attention_block(64, 128, 64)
        self.att3 = Attention_block(64, 128, 64)
        self.att4 = Attention_block(64, 128, 64)

        self.refine4 = nn.Sequential(
            nn.Conv3d(320, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()

        )
        self.refine3 = nn.Sequential(
            nn.Conv3d(320, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.refine2 = nn.Sequential(
            nn.Conv3d(320, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv3d(320, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.pre = nn.Conv3d(64, 1, kernel_size=1)
        self.pre11 = nn.Conv3d(16, 1, kernel_size=1)
        self.refine = nn.Sequential(nn.Conv3d(256, 64, kernel_size=1),
                                    nn.GroupNorm(32, 64),
                                    nn.PReLU(), )

        self.predict = nn.Conv3d(64, 32, kernel_size=1)
        self.predict11 = nn.Conv3d(32, 1, kernel_size=1)
        self.predict1 = nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear')







    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())



    def forward(self, x):

        out = self.conv3d_c1_1(x)
        # print(out.shape)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        # print(context_1.shape)
        out = self.inorm3d_c1(out)
        out1 = self.lrelu(out)


        out = self.conv3d_c2(out1)
        uu1 = self.uu1(out)
        c11 = torch.cat([uu1, out1], dim=1)
        t1 = self.cc1(c11)
        # print(out.shape)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out2 = self.lrelu(out)
        context_2 = out


        out = self.conv3d_c3(out)
        uu2 = self.uu2(out)
        c22 = torch.cat([uu2, out2], dim=1)
        t2 = self.cc2(c22)
        # print(out.shape)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out3 = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        uu3 = self.uu3(out)
        c33 = torch.cat([uu3, out3], dim=1)
        t3 = self.cc3(c33)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out4 = self.lrelu(out)
        # print(out.shape)
        context_4 = out


        out = self.conv3d_c5(out)
        uu4 = self.uu4(out)
        # print(uu4.shape)
        c44 = torch.cat([uu4, out4], dim=1)
        t4 = self.cc4(c44)
        # print(t4.shape)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        # print(out.shape)

        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        # print(out.shape)
        # out = self.conv3d_l0(out)
        # print(out.shape)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)


        out = torch.cat([out, t4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        ds1 = out
        d1 = self.p1(ds1)
        # print("ds1:",ds1.shape)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)


        out = torch.cat([out, t3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out

        # print("ds2:",ds2.shape)
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)


        out = torch.cat([out, t2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        d3 = self.p3(ds3)
        # print("ds3:",ds3.shape)
        out = self.conv3d_l3(out)
        # print(out.shape)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        # print(out.shape)
        # Level 4 localization pathway
        out = torch.cat([out, t1], dim=1)
        # print(out.shape)
        out = self.conv_norm_lrelu_l4(out)
        # print(out.shape)
        out_pred = self.conv3d_l4(out)
        ds4 =out_pred
        d4 = self.p4(ds4)
        # print("out_pred  ",out_pred.shape)
        ds1_1x1_conv =self.ds1_1x1_conv3d(ds1)
        ds1_up = self.upsacle2(ds1_1x1_conv)
        # print(ds1_up.shape)
        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds2_up = self.upsacle1(ds2_1x1_conv)
        # print(ds2_up.shape)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds3up = self.upsacle(ds3_1x1_conv)
        # print(ds3up.shape)
        ds1_2_3 = ds3up + ds2_up+ds1_up
        out = out_pred +ds1_2_3
        # print(out.shape)
        out = self.output4(out)
        # print(out.shape)
        ds1_1x1_conv = self.pre11(ds1_1x1_conv)
        ds2_1x1_conv = self.pre11(ds2_1x1_conv)
        ds3_1x1_conv = self.pre11(ds3_1x1_conv)

        d11 = F.upsample(ds1_1x1_conv, size=out.size()[2:], mode='trilinear')
        d22 = F.upsample(ds2_1x1_conv, size=out.size()[2:], mode='trilinear')
        d33 = F.upsample(ds3_1x1_conv, size=out.size()[2:], mode='trilinear')

        p1= F.upsample(d1, size=ds2.size()[2:], mode='trilinear')
        p2 = ds2
        p3 = F.upsample(d3, size=ds2.size()[2:], mode='trilinear')
        p4 = F.upsample(d4, size=ds2.size()[2:], mode='trilinear')

        fuse1 = self.fuse1(torch.cat((p1, p2, p3, p4), 1))

        attention4 = self.att4(fuse1, p4)
        attention3 = self.att3(fuse1, p3)
        attention2 = self.att2(fuse1, p2)
        attention1 = self.att1(fuse1, p1)

        refine4 = self.refine4(torch.cat((p4, torch.cat((attention4, fuse1), 1)), 1))
        refine3 = self.refine3(torch.cat((p3, torch.cat((attention3, fuse1), 1)), 1))
        refine2 = self.refine2(torch.cat((p2, torch.cat((attention2, fuse1), 1)), 1))
        refine1 = self.refine1(torch.cat((p1, torch.cat((attention1, fuse1), 1)), 1))

        refine = self.refine(torch.cat((refine1, refine2, refine3, refine4), 1))

        predict = self.predict(refine)
        predict = self.predict11(predict)
        predict = self.predict1(predict)

        predict = nn.Sigmoid()(predict)

        return predict


import torch.nn as nn
import torch
import os
import torch.nn.functional as F


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi

        return out


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        rate_list = (1, rate, rate)
        self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate_list, dilation=rate_list)
        self.group_norm = nn.GroupNorm(32, planes)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.group_norm(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Modified3DUNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=16):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.3)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsacle1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsacle2 = nn.Upsample(scale_factor=8, mode='nearest')
        self.softmax = nn.Softmax(dim=1)
        self.output4 = nn.Sequential(  # decoder2的输出
            nn.Conv3d(16, 1, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )
        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)

        self.ds1_1x1_conv3d = nn.Conv3d(self.base_n_filter * 16, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)

        self.p1 = nn.Conv3d(256, 128, kernel_size=1, stride=1, padding=0,
                            bias=False)
        self.p3 = nn.Conv3d(64, 128, kernel_size=1, stride=1, padding=0,
                            bias=False)
        self.p4 = nn.Conv3d(16, 128, kernel_size=1, stride=1, padding=0,
                            bias=False)

        self.fuse1 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.att1 = Attention_block(64, 128, 64)
        self.att2 = Attention_block(64, 128, 64)
        self.att3 = Attention_block(64, 128, 64)
        self.att4 = Attention_block(64, 128, 64)

        # self.attention4 = nn.Sequential(
        #     nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
        #     nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
        #     nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        # )
        #
        # self.attention3 = nn.Sequential(
        #     nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
        #     nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
        #     nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        # )
        # self.attention2 = nn.Sequential(
        #     nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
        #     nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
        #     nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        # )
        # self.attention1 = nn.Sequential(
        #     nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
        #     nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
        #     nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        # )
        self.refine4 = nn.Sequential(
            nn.Conv3d(320, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()

        )
        self.refine3 = nn.Sequential(
            nn.Conv3d(320, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.refine2 = nn.Sequential(
            nn.Conv3d(320, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv3d(320, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.pre = nn.Conv3d(64, 1, kernel_size=1)
        self.pre11 = nn.Conv3d(16, 1, kernel_size=1)
        self.refine = nn.Sequential(nn.Conv3d(256, 64, kernel_size=1),
                                    nn.GroupNorm(32, 64),
                                    nn.PReLU(), )

        rates = (1, 6, 12, 18)
        self.aspp1 = ASPP_module(64, 64, rate=rates[0])
        self.aspp2 = ASPP_module(64, 64, rate=rates[1])
        self.aspp3 = ASPP_module(64, 64, rate=rates[2])
        self.aspp4 = ASPP_module(64, 64, rate=rates[3])
        self.aspp_conv = nn.Conv3d(256, 64, 1)
        self.aspp_gn = nn.GroupNorm(32, 64)
        self.predict = nn.Conv3d(64, 1, kernel_size=1)
        self.predict1 = nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear')

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        # print(out.shape)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        # print(out.shape)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        # print(out.shape)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        # print(out.shape)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        # print(out.shape)

        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        # print(out.shape)
        # out = self.conv3d_l0(out)
        # print(out.shape)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        ds1 = out
        d1 = self.p1(ds1)
        # print("ds1:",ds1.shape)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out

        # print("ds2:",ds2.shape)
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        d3 = self.p3(ds3)
        # print("ds3:",ds3.shape)
        out = self.conv3d_l3(out)
        # print(out.shape)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        # print(out.shape)
        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        # print(out.shape)
        out = self.conv_norm_lrelu_l4(out)
        # print(out.shape)
        out_pred = self.conv3d_l4(out)
        ds4 = out_pred
        d4 = self.p4(ds4)
        # print("out_pred  ",out_pred.shape)
        ds1_1x1_conv = self.ds1_1x1_conv3d(ds1)
        ds1_up = self.upsacle2(ds1_1x1_conv)
        # print(ds1_up.shape)
        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds2_up = self.upsacle1(ds2_1x1_conv)
        # print(ds2_up.shape)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds3up = self.upsacle(ds3_1x1_conv)
        # print(ds3up.shape)
        ds1_2_3 = ds3up + ds2_up + ds1_up
        out = out_pred + ds1_2_3
        # print(out.shape)
        out = self.output4(out)
        # print(out.shape)
        ds1_1x1_conv = self.pre11(ds1_1x1_conv)
        ds2_1x1_conv = self.pre11(ds2_1x1_conv)
        ds3_1x1_conv = self.pre11(ds3_1x1_conv)
        # print(ds1_1x1_conv.shape)
        d11 = F.upsample(ds1_1x1_conv, size=out.size()[2:], mode='trilinear')
        d22 = F.upsample(ds2_1x1_conv, size=out.size()[2:], mode='trilinear')
        d33 = F.upsample(ds3_1x1_conv, size=out.size()[2:], mode='trilinear')
        # print("d11", d11.shape)
        # print("d22", d22.shape)
        # print("d33", d33.shape)
        # d44 = F.upsample(out_pred, size=out.size()[2:], mode='trilinear')
        # print(d44.shape)
        p1 = F.upsample(d1, size=ds2.size()[2:], mode='trilinear')
        p2 = ds2
        p3 = F.upsample(d3, size=ds2.size()[2:], mode='trilinear')
        p4 = F.upsample(d4, size=ds2.size()[2:], mode='trilinear')
        # print("ds3333333", p3.shape)
        # print("ds2222222", p2.shape)
        # print("ds1111111", p1.shape)
        # print("ds4444444", p4.shape)
        fuse1 = self.fuse1(torch.cat((p1, p2, p3, p4), 1))
        # print("fuse1", fuse1.shape)
        attention4 = self.att4(fuse1, p4)
        attention3 = self.att3(fuse1, p3)
        attention2 = self.att2(fuse1, p2)
        attention1 = self.att1(fuse1, p1)
        # print(attention1.shape)
        # attention4 = self.attention4(torch.cat((p1, fuse1), 1))
        # attention3 = self.attention3(torch.cat((p2, fuse1), 1))
        # attention2 = self.attention2(torch.cat((p3, fuse1), 1))
        # attention1 = self.attention1(torch.cat((p4, fuse1), 1))
        refine4 = self.refine4(torch.cat((p4, torch.cat((attention4, fuse1), 1)), 1))
        refine3 = self.refine3(torch.cat((p3, torch.cat((attention3, fuse1), 1)), 1))
        refine2 = self.refine2(torch.cat((p2, torch.cat((attention2, fuse1), 1)), 1))
        refine1 = self.refine1(torch.cat((p1, torch.cat((attention1, fuse1), 1)), 1))
        pre1 = self.pre(refine4)
        pre2 = self.pre(refine4)
        pre3 = self.pre(refine4)
        pre4 = self.pre(refine4)
        # print(pre1.shape)
        pre1 = F.upsample(pre1, size=out.size()[2:], mode='trilinear')
        pre2 = F.upsample(pre2, size=out.size()[2:], mode='trilinear')
        pre3 = F.upsample(pre3, size=out.size()[2:], mode='trilinear')
        pre4 = F.upsample(pre4, size=out.size()[2:], mode='trilinear')

        refine = self.refine(torch.cat((refine1, refine2, refine3, refine4), 1))

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)
        aspp = torch.cat((aspp1, aspp2, aspp3, aspp4), dim=1)

        aspp = self.aspp_gn(self.aspp_conv(aspp))

        predict = self.predict(aspp)
        predict = self.predict1(predict)
        d11 = nn.Sigmoid()(d11)
        d22 = nn.Sigmoid()(d22)
        d33 = nn.Sigmoid()(d33)
        pre1 = nn.Sigmoid()(pre1)
        pre2 = nn.Sigmoid()(pre2)
        pre3 = nn.Sigmoid()(pre3)
        pre4 = nn.Sigmoid()(pre4)
        predict = nn.Sigmoid()(predict)

        return predict


