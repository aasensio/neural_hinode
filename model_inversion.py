import torch
import torch.nn as nn
import torch.utils.data

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, use_activation=True):
        super(ConvBlock, self).__init__()

        self.use_activation = use_activation
        
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=1)
        
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

        self.reflection = nn.ReflectionPad2d(int((kernel_size-1)/2))
        
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):        
        out = self.reflection(x)
        out = self.conv(out)
        if (self.use_activation):
            out = self.elu(out)       
            
        return out

class block(nn.Module):
    def __init__(self, in_planes, out_planes=7):
        super(block, self).__init__()        
        
        self.C00 = ConvBlock(in_planes, 128, kernel_size=3, use_activation=True)
        self.C01 = ConvBlock(128, 128, kernel_size=3, use_activation=True)
        self.C02 = ConvBlock(128, 128, kernel_size=3, use_activation=True)
        self.C03 = ConvBlock(128, out_planes, kernel_size=3, use_activation=False)

        # self.C10 = ConvBlock(in_planes, 32, kernel_size=3, use_activation=True)
        # self.C11 = ConvBlock(32, 32, kernel_size=3, use_activation=True)
        # self.C12 = ConvBlock(32, 32, kernel_size=3, use_activation=True)
        # self.C13 = ConvBlock(32, out_planes, kernel_size=3, use_activation=False)

        # self.C20 = ConvBlock(in_planes, 32, kernel_size=3, use_activation=True)
        # self.C21 = ConvBlock(32, 32, kernel_size=3, use_activation=True)
        # self.C22 = ConvBlock(32, 32, kernel_size=3, use_activation=True)
        # self.C23 = ConvBlock(32, out_planes, kernel_size=3, use_activation=False)

        # self.C30 = ConvBlock(in_planes, 32, kernel_size=3, use_activation=True)
        # self.C31 = ConvBlock(32, 32, kernel_size=3, use_activation=True)
        # self.C32 = ConvBlock(32, 32, kernel_size=3, use_activation=True)
        # self.C33 = ConvBlock(32, out_planes, kernel_size=3, use_activation=False)

        # self.C40 = ConvBlock(in_planes, 32, kernel_size=3, use_activation=True)
        # self.C41 = ConvBlock(32, 32, kernel_size=3, use_activation=True)
        # self.C42 = ConvBlock(32, 32, kernel_size=3, use_activation=True)
        # self.C43 = ConvBlock(32, out_planes, kernel_size=3, use_activation=False)

        # self.C50 = ConvBlock(in_planes, 32, kernel_size=3, use_activation=True)
        # self.C51 = ConvBlock(32, 32, kernel_size=3, use_activation=True)
        # self.C52 = ConvBlock(32, 32, kernel_size=3, use_activation=True)
        # self.C53 = ConvBlock(32, out_planes, kernel_size=3, use_activation=False)

        # self.C60 = ConvBlock(in_planes, 32, kernel_size=3, use_activation=True)
        # self.C61 = ConvBlock(32, 32, kernel_size=3, use_activation=True)
        # self.C62 = ConvBlock(32, 32, kernel_size=3, use_activation=True)
        # self.C63 = ConvBlock(32, 7, kernel_size=3, use_activation=False)               
                
    def forward(self, x):

        tmp0 = self.C00(x)
        tmp1 = self.C01(tmp0)
        out = tmp0 + tmp1
        out = self.C02(out)
        out0 = self.C03(out)

        # tmp0 = self.C10(x)
        # tmp1 = self.C11(tmp0)
        # out = tmp0 + tmp1
        # out = self.C12(out)
        # out1 = self.C13(out)

        # tmp0 = self.C20(x)
        # tmp1 = self.C21(tmp0)
        # out = tmp0 + tmp1
        # out = self.C22(out)
        # out2 = self.C23(out)

        # tmp0 = self.C30(x)
        # tmp1 = self.C31(tmp0)
        # out = tmp0 + tmp1
        # out = self.C32(out)
        # out3 = self.C33(out)

        # tmp0 = self.C40(x)
        # tmp1 = self.C41(tmp0)
        # out = tmp0 + tmp1
        # out = self.C42(out)
        # out4 = self.C43(out)

        # tmp0 = self.C50(x)
        # tmp1 = self.C51(tmp0)
        # out = tmp0 + tmp1
        # out = self.C52(out)
        # out5 = self.C53(out)

        # tmp0 = self.C60(x)
        # tmp1 = self.C61(tmp0)
        # out = tmp0 + tmp1
        # out = self.C62(out)
        # out6 = self.C63(out)

        # out = torch.cat([out0, out1, out2, out3, out4, out5, out6], dim=1)
        
        return out0