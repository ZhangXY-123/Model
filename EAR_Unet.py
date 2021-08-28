import torch
import torch.nn as nn
import torch.nn.functional as F
from pp.EfficientNet import EfficientNet
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            # nn.GroupNorm(32, out_ch),
        )


    def forward(self, input):
        x1 = self.conv(input)
        x2 = self.channel_conv(input)
        x = x1 + x2
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class Unet(nn.Module):
    def __init__(self, pretrained_net, out_ch):
        super(Unet, self).__init__()
        # print("EfficientUnet_git_b6_res")
        self.pretrained_net = pretrained_net
        self.up6 = nn.ConvTranspose2d(1792, 512, 2, stride=2)
        self.att1 =Attention_block(512, 160, 160)
        self.conv6 = DoubleConv(160 + 512, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att2 = Attention_block(256, 56, 56)
        self.conv7 = DoubleConv(256 + 56, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 =Attention_block(128, 32, 32)
        self.conv8 = DoubleConv(128 + 32, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att4 = Attention_block(64, 24, 24)
        self.conv9 = DoubleConv(64 + 24, 64)
        self.up10 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att5 = Attention_block(32, 1, 1)
        self.conv10 = DoubleConv(33, 32)
        self.conv11 = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)

        x1 = output['x1']
        x2 = output['x2']
        x3 = output['x3']
        x4 = output['x4']
        x5 = output['x5']

        up_6 = self.up6(x5)
        x4 = self.att1(up_6, x4)
        merge6 = torch.cat([up_6, x4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        x3 = self.att2(up_7, x3)
        merge7 = torch.cat([up_7, x3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        x2 = self.att3(up_8, x2)
        merge8 = torch.cat([up_8, x2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        x1 = self.att4(up_9, x1)
        merge9 = torch.cat([up_9, x1], dim=1)
        c9 = self.conv9(merge9)
        up_10 = self.up10(c9)
        x = self.att5(up_10, x)
        merge10 = torch.cat([up_10, x], dim=1)
        c10 = self.conv10(merge10)
        c11 = self.conv11(c10)
        out = nn.Sigmoid()(c11)
        return out
if __name__ == "__main__":
    efficient_model = EfficientNet()
    model = Unet(out_ch=1, pretrained_net=efficient_model)
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    print(output.shape)