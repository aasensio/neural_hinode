import torch
import torch.nn as nn
import torch.utils.data
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, upsample=False, use_bn=True, use_activation=True):
        super(ConvBlock, self).__init__()

        self.use_bn = use_bn
        self.use_activation = use_activation

        self.upsample = upsample

        if (upsample):
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=1)
        else:
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride)

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

        self.reflection = nn.ReflectionPad2d(int((kernel_size-1)/2))

        if (self.use_bn):
            self.bn = nn.BatchNorm2d(inplanes)

        if (self.use_activation):
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if (self.use_bn):
            out = self.bn(x)
            if (self.use_activation):
                out = self.relu(out)
            
            if (self.upsample):
                out = torch.nn.functional.interpolate(out, scale_factor=2)

            out = self.reflection(out)
            out = self.conv(out)

        else:
            out = self.reflection(x)
            out = self.conv(out)
            if (self.use_activation):
                out = self.relu(out)                    
            
        return out

class block(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(block, self).__init__()        
        self.A01 = ConvBlock(in_planes, 32, kernel_size=3, use_bn=True, use_activation=False)
        
        self.C01 = ConvBlock(32, 64, stride=2)
        self.C02 = ConvBlock(64, 64)
        self.C03 = ConvBlock(64, 64)
        self.C04 = ConvBlock(64, 64, kernel_size=1)

        self.C11 = ConvBlock(64, 64)
        self.C12 = ConvBlock(64, 64)
        self.C13 = ConvBlock(64, 64)
        self.C14 = ConvBlock(64, 64, kernel_size=1)
        
        self.C21 = ConvBlock(64, 128, stride=2)
        self.C22 = ConvBlock(128, 128)
        self.C23 = ConvBlock(128, 128)
        self.C24 = ConvBlock(128, 128, kernel_size=1)
        
        self.C31 = ConvBlock(128, 256, stride=2)
        self.C32 = ConvBlock(256, 256)
        self.C33 = ConvBlock(256, 256)
        self.C34 = ConvBlock(256, 256, kernel_size=1)
        
        self.C41 = ConvBlock(256, 128, upsample=True)
        self.C42 = ConvBlock(128, 128)
        self.C43 = ConvBlock(128, 128)
        self.C44 = ConvBlock(128, 128)
        
        self.C51 = ConvBlock(128, 64, upsample=True)
        self.C52 = ConvBlock(64, 64)
        self.C53 = ConvBlock(64, 64)
        self.C54 = ConvBlock(64, 64)
        
        self.C61 = ConvBlock(64, 64, upsample=True)
        self.C62 = ConvBlock(64, 64)
        self.C63 = ConvBlock(64, 64)

        self.C64 = nn.Conv2d(64, out_planes, kernel_size=1, stride=1)
        nn.init.kaiming_normal_(self.C64.weight)
        nn.init.constant_(self.C64.bias, 0.1)

        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x):

        A01 = self.A01(x)

        # N -> N/2
        C01 = self.C01(A01)
        C02 = self.C02(C01)
        C03 = self.C03(C02)
        C04 = self.C04(C03)
        C04 += C01
        
        # N/2 -> N/2
        C11 = self.C11(C04)
        C12 = self.C12(C11)
        C13 = self.C13(C12)
        C14 = self.C14(C13)
        C14 += C11
        
        # N/2 -> N/4
        C21 = self.C21(C14)
        C22 = self.C22(C21)
        C23 = self.C23(C22)
        C24 = self.C24(C23)
        C24 += C21
        
        # N/4 -> N/8
        C31 = self.C31(C24)
        C32 = self.C32(C31)
        C33 = self.C33(C32)
        C34 = self.C34(C33)
        C34 += C31
        
        C41 = self.C41(C34)
        C41 += C24
        C42 = self.C42(C41)
        C43 = self.C43(C42)
        C44 = self.C44(C43)
        C44 += C41
        
        C51 = self.C51(C44)
        C51 += C14
        C52 = self.C52(C51)
        C53 = self.C53(C52)
        C54 = self.C54(C53)
        C54 += C51
        
        C61 = self.C61(C54)        
        C62 = self.C62(C61)
        C63 = self.C63(C62)
        C64 = self.C64(C63)

        # Avoid sigmoids that 
        C64 = torch.clamp(C64, -16, 8)

        C64 = self.sigmoid(C64)

        tmp = C64.view(C64.size(0), 2, C64.size(1)//2, C64.size(2), C64.size(3))

        # Use Box-Muller transform to generate outputs that like on a Gaussian ball as assumed in the embedding

        tmp1 = torch.sqrt(-2.0 * torch.log(tmp[:,0,:,:,:])) * torch.cos(2.0 * np.pi * tmp[:,1,:,:,:])
        tmp2 = torch.sqrt(-2.0 * torch.log(tmp[:,0,:,:,:])) * torch.sin(2.0 * np.pi * tmp[:,1,:,:,:])

        out = torch.cat([tmp1, tmp2], dim=1)
                
        return out