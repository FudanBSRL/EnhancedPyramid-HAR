import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_weights

class ChannelAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.BN = nn.BatchNorm1d(1)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) + self.maxpool(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        # y=self.BN(y)
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return y


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()

        self.compress = FCSPool()

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_compress = self.compress(x)
        scale = self.sigmoid(x_compress)
        return x * scale

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        # self.relu = nn.ReLU() if relu else None
        self.relu = nn.GELU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FCSPool(nn.Module):
    def __init__(self):
        super(FCSPool, self).__init__()
        self.channel_attention = ChannelAttention()
        kernel_size = (5, 1)
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(2, 0), relu=False)
        self.BN = nn.BatchNorm2d(1)
        self.RELU = nn.ReLU()
        self.w1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        channel_attention_weights = self.channel_attention(x)

        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, 1).unsqueeze(1)

        # 沿通道轴拼接最大池化和平均池化结果
        output = torch.cat((max_pool, avg_pool), dim=1)
        output = self.conv(output)
        output = self.BN(output)
        output = self.RELU(output)
        output = self.w1 * channel_attention_weights + self.w2 * output
        # print("channel_attention_weights {} , out put.shape {} ".format(channel_attention_weights.shape, output.shape))
        return output


class TripletAttention(nn.Module):
    def __init__(self,input, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
        self.weights = nn.Parameter(torch.randn(3), requires_grad=True)
        self.u1 = nn.Parameter(torch.tensor(1/3), requires_grad=True)  # 定义学习参数w1，并初始化为0.5
        self.u2 = nn.Parameter(torch.tensor(1/3), requires_grad=True)  # 定义学习参数w2，并初始化为0.5
        self.u3 = nn.Parameter(torch.tensor(1/3), requires_grad=True) # 定义学习参数w2，并初始化为0.5
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()


        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = (self.u1 * x_out + self.u2 * x_out11 + self.u3 * x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out


class ResConvNet(nn.Module):
    def __init__(self, in_planes, s, w):
        super(ResConvNet, self).__init__()
        self.s = s
        self.w = w
        #考虑整除的情况
        self.channel = in_planes // self.s
        self.module_list = nn.ModuleList()

        for i in range(0, self.s):
            if i == 0:
                self.module_list.append(self.Channel_Conv_BN(in_planes//s, w, kernel_size = (2*i+1,1), padding= (i,0)))
            else:
                self.module_list.append(self.Channel_Conv_BN(in_planes//s + w, w, kernel_size = (2*i+1,1), padding= (i,0)))

    def Channel_Conv_BN(self, in_ch, out_ch, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0)):
        Channel_Conv_BN = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return Channel_Conv_BN
    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(0, self.s):
            if i == 0:
                y = self.module_list[i](x[i])
                out = y
            else:
                x[i] = torch.cat((y , x[i]),dim=1)
                y = self.module_list[i](x[i])
                out = torch.cat((out, y),dim=1)
        return out

class Sym_ResConvNet(nn.Module):
    def __init__(self, in_planes, s, w):
        super(Sym_ResConvNet, self).__init__()
        self.s = s
        self.w = w
        #考虑整除的情况
        self.channel = in_planes // self.s
        self.module_list = nn.ModuleList()

        for i in range(0, self.s):
            if i == self.s-1:
                self.module_list.append(self.Channel_Conv_BN(in_planes//s, w, kernel_size = (2*i+1,1), padding= (i,0)))
            else:
                self.module_list.append(self.Channel_Conv_BN(in_planes//s + w, w, kernel_size = (2*i+1,1), padding= (i,0)))

    def Channel_Conv_BN(self, in_ch, out_ch, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)):
        Channel_Conv_BN = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return Channel_Conv_BN
    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(self.s-1,-1,-1):
            if i == self.s-1:
                y = self.module_list[i](x[i])
                out = y
            else:
                x[i] = torch.cat((x[i], y),dim=1)
                y = self.module_list[i](x[i])
                out = torch.cat((y,out),dim=1)
        return out

class Pry_ResConvNet(nn.Module):
    def __init__(self, in_planes, s, w):
        super(Pry_ResConvNet, self).__init__()
        self.s = s
        self.w = w
        self.Sym_ResConvNet = Sym_ResConvNet(in_planes, self.s, self.w)
        self.ResConvNet = ResConvNet(in_planes, self.s, self.w)
        self.ConvBN = nn.Sequential(
            nn.Conv2d((self.w*self.s)*2, in_planes, kernel_size=(1,1),stride=(1,1),padding=(0,0)),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )

    def forward(self, x):
        out_ori = self.Sym_ResConvNet(x)
        out_sym = self.ResConvNet(x)
        out = torch.cat((out_ori,out_sym),dim=1)
        out = self.ConvBN(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding,s, w):
        super(BasicBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, kernel_size, (1,1), padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
            nn.GELU()
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )
        self.symRes = self._make_layer(output_channel,s,output_channel//w)

    def _make_layer(self, in_planes, s, w):
        return Pry_ResConvNet(in_planes, s, w)

    def forward(self, x):
        identity = self.shortcut(x)
        # print(identity.shape)

        x = self.layer1(x)
        # print(x.shape)
        # x = self.layer2(x)
        x = self.symRes(x)
        x = x + identity
        x = F.gelu(x)
        return x


class Pry_ResNet(nn.Module):
    def __init__(self, input_channel, num_classes, s, w):
        super(Pry_ResNet, self).__init__()

        self.layer1 = self._make_layers(input_channel, 32, (5,1), (3,1), (3,0), s, w)
        self.layer2 = self._make_layers(32, 64, (5,1), (3,1), (3,0), s, w)
        self.layer3 = self._make_layers(64, 128, (5,1), (3,1), (3,0), s, w)
        self.fc = nn.Linear(9216, num_classes)
        self.triplet_attention = self.make_att(input)

    def make_att(self, input):
        return TripletAttention(input)
    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding, s, w):
        return BasicBlock(input_channel, output_channel, kernel_size, stride, padding, s, w)

    def forward(self, x):
        x = self.layer1(x)
        x = self.triplet_attention(x)
        x = self.layer2(x)
        x = self.triplet_attention(x)
        x = self.layer3(x)
        x = self.triplet_attention(x)
        x = F.max_pool2d(x, (4,1))
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

import torch
from pry_resnet import Pry_ResNet
from thop import profile
from torchsummary import summary

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pry_ResNet(1, 12, 4, 8).to(device)

    input_tensor = torch.randn(1, 1, 171, 36).to(device)
    flops, params = profile(model, inputs=(input_tensor, ))

    print('flops: %.4f M, params: %.4f M' % (flops / 1000000.0, params / 1000000.0))
