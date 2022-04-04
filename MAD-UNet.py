import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from att import Attention_block
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


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    import time
    from thop import profile
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Modified3DUNet(1, 16)
    params = sum(param.numel() for param in model.parameters())
    print(params)
    input = torch.randn(1, 1, 16, 256, 256)  # BCDHW
    input = input.to(device)
    out = model(input)

    macs, params = profile(model, inputs=(input,), )

    print(macs)
    # print(out)
    end = time.time()
    print(end - start)
    print("output.shape:", out.shape)