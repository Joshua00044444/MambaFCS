"""
ResBlock / SqueezeExcitation —— 残差块 + SE 注意力
======================================================
用途：在 ChangeDecoder / SemanticDecoder 中作为金字塔各级的
"smooth_layer_*" 平滑模块（对融合后的特征做一次轻量残差精修）。

ResBlock：标准 ResNet 残差块（两层 3x3 卷积 + BN + ReLU），
          并在残差路径末尾插入 SE 注意力做通道加权。
SqueezeExcitation：经典 SE 模块 —— 全局平均池化 + 双全连接退频/升频
          + sigmoid，对每个通道重加权（重要通道保留、次要通道压制）。
"""
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        # 第一层 3x3 卷积：负责通道转换（或保持）与空间缩小（stride>1 时）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二层 3x3 卷积：保持通道，提取更细语义（步长恒为 1）
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # downsample：若输入/输出通道或尺寸不一致，用它把恒等分支对齐
        #（本项目调用时均未传入，即 in==out、stride==1，恒等分支直接复用输入）
        self.downsample = downsample

        # SE 注意力附加在残差路径上（输出通道上做通道重加权）
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 通道注意力：先 squeeze，再 excitation，得到每个通道的 (0,1) 权重
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差相加 + 激活（标准 ResNet 写法）
        out += identity
        out = self.relu(out)

        return out

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # squeeze：全局平均池化，把 (B,C,H,W) 压成 (B,C,1,1)，得到通道描述子
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # excitation：两层全连接（先缩后升），学习通道之间的相关性，
        # 输出 sigmoid 后即每个通道的注意力权重
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)  # 恢复成通道权重图 (B,C,1,1)
        return x * y.expand_as(x)                 # 逐通道加权（广播到 H,W）
