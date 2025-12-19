import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ResNet18Saliency(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载预训练ResNet18并拆分编码器
        resnet = models.resnet18(pretrained=pretrained)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64通道, 1/2
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 64通道, 1/4
        self.encoder3 = resnet.layer2  # 128通道, 1/8
        self.encoder4 = resnet.layer3  # 256通道, 1/16
        self.encoder5 = resnet.layer4  # 512通道, 1/32

        # 解码器：改为上采样+卷积，消除棋盘伪影
        # feat5: 512, 1/32 -> up -> 512, 1/16
        self.decoder5_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder5_conv = DecoderBlock(512, 256)

        # fuse4: 256 (dec) + 256 (enc) = 512 -> 256
        self.decoder4_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder4_conv = DecoderBlock(256 + 256, 128)

        # fuse3: 128 (dec) + 128 (enc) = 256 -> 64
        self.decoder3_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder3_conv = DecoderBlock(128 + 128, 64)

        # fuse2: 64 (dec) + 64 (enc) = 128 -> 64
        self.decoder2_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder2_conv = DecoderBlock(64 + 64, 64)

        # fuse1: 64 (dec) + 64 (enc) = 128 -> 32
        self.decoder1_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder1_conv = DecoderBlock(64 + 64, 32)
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器提取多尺度特征
        feat1 = self.encoder1(x)      # 1/2
        feat2 = self.encoder2(feat1)  # 1/4
        feat3 = self.encoder3(feat2)  # 1/8
        feat4 = self.encoder4(feat3)  # 1/16
        feat5 = self.encoder5(feat4)  # 1/32

        # 解码器融合与上采样
        # 5 -> 4
        dec5 = self.decoder5_up(feat5)
        dec5 = self.decoder5_conv(dec5)
        
        # 4 -> 3
        if dec5.size()[2:] != feat4.size()[2:]:
             dec5 = F.interpolate(dec5, size=feat4.size()[2:], mode='bilinear', align_corners=True)
             
        fuse4 = torch.cat([dec5, feat4], dim=1)
        dec4 = self.decoder4_up(fuse4)
        dec4 = self.decoder4_conv(dec4)

        # 3 -> 2
        if dec4.size()[2:] != feat3.size()[2:]:
             dec4 = F.interpolate(dec4, size=feat3.size()[2:], mode='bilinear', align_corners=True)

        fuse3 = torch.cat([dec4, feat3], dim=1)
        dec3 = self.decoder3_up(fuse3)
        dec3 = self.decoder3_conv(dec3)

        # 2 -> 1
        if dec3.size()[2:] != feat2.size()[2:]:
             dec3 = F.interpolate(dec3, size=feat2.size()[2:], mode='bilinear', align_corners=True)

        fuse2 = torch.cat([dec3, feat2], dim=1)
        dec2 = self.decoder2_up(fuse2)
        dec2 = self.decoder2_conv(dec2)

        # 1 -> 0
        if dec2.size()[2:] != feat1.size()[2:]:
             dec2 = F.interpolate(dec2, size=feat1.size()[2:], mode='bilinear', align_corners=True)
             
        fuse1 = torch.cat([dec2, feat1], dim=1)
        dec1 = self.decoder1_up(fuse1)
        dec1 = self.decoder1_conv(dec1)
        
        out = self.final_conv(dec1)
        
        # 问题出在这里：encoder1输出是1/2，decoder恢复到1/2，然后上采样2倍回到原图。
        # 但是，ResNet的Conv1 stride=2 -> 1/2.
        # Encoder1 (conv1+pool) is actually:
        # conv1(s=2) -> 1/2
        # maxpool(s=2) -> 1/4 (feat2)
        # So feat1 is 1/2 size.
        
        # Let's trace sizes for 320x320:
        # x: 320
        # feat1: 160 (1/2)
        # feat2: 80 (1/4)
        # feat3: 40 (1/8)
        # feat4: 20 (1/16)
        # feat5: 10 (1/32)
        
        # dec5 (up from 10): 20. fuse with feat4(20). -> 20
        # dec4 (up from 20): 40. fuse with feat3(40). -> 40
        # dec3 (up from 40): 80. fuse with feat2(80). -> 80
        # dec2 (up from 80): 160. fuse with feat1(160). -> 160
        # dec1 (up from 160): 320. fuse with ?? NO.
        # 
        # Wait, code says:
        # fuse1 = torch.cat([dec2, feat1], dim=1) 
        # dec2 is upsampled output of loop 2->1. Let's re-read carefully.
        
        # Loop 5->4: dec5 (from feat5) fuses feat4. Output dec4 size of feat4 (1/16).
        # Loop 4->3: dec4 fuses feat3. Output dec3 size of feat3 (1/8).
        # Loop 3->2: dec3 fuses feat2. Output dec2 size of feat2 (1/4).
        # Loop 2->1: dec2 fuses feat1. Output dec1 size of feat1 (1/2).
        
        # Current code:
        # fuse1 = torch.cat([dec2, feat1], dim=1) -> dec2 comes from fuse2 (dec3+feat2).
        # dec2 = self.decoder2_up(fuse2) -> upsamples feat2 size (1/4) to 1/2. Matches feat1.
        # So fuse1 is size 1/2 (160).
        # dec1 = self.decoder1_up(fuse1) -> Upsamples fuse1 (1/2) to (1/1). (320).
        # dec1 = self.decoder1_conv(dec1) -> Conv on 320.
        
        # out = self.final_conv(dec1) -> 320.
        # THEN: out = F.interpolate(out, scale_factor=2) -> 640!
        
        # ERROR FOUND: My previous edit added an extra interpolate at the end because I thought output was 1/2.
        # But looking at the chain:
        # decoder1_up upsamples fuse1 (which is 1/2 size) to full size.
        # So 'out' is already full size.
        # Removing the final interpolate.
        
        return self.sigmoid(out)
