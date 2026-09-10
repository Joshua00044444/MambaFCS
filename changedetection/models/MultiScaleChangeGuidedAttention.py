"""
MultiScaleChangeGuidedAttention —— 变化引导注意力（CGA）的三种实现
===================================================================
本文件是论文"变化引导语义精修"的候选模块集合，包含 3 个版本：

1. MultiScaleChangeGuidedAttention：多尺度变体 —— 每个 level 一个 1x1 卷积
   把 change_map 投影到该 level 的通道数，再 sigmoid，做"加性门控" 特征×(1+att)。
2. MultiScaleChangeGuidedAttention_StageByStage：逐级变体 —— 与 1 类似，
   但输入的是逐级 change_maps（每个 level 独立一张），同样做"加性门控"。
3. ChangeGuidedAttention：最终实际使用的版本（SemanticDecoder 依赖）——
   无卷积投影，直接 特征 × sigmoid(change_map) 的纯乘法门控。

注意：版本 1、2 在本项目中**未被任何模型引用**（遗留的候选实现）；
真正生效的是版本 3。三者对比如下：
    版本 1、2：feature * (1 + attention) —— 增强式门控（attention≈0 时特征原样保留）
    版本 3  ：feature * attention        —— 乘法门控（attention≈0 时特征被抑制）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleChangeGuidedAttention(nn.Module):
    """多尺度变体：每级用独立的 1x1 conv 把变化图投影到该级特征通道数。"""
    def __init__(self, channels_list):
        super().__init__()
        # channels_list：各尺度特征对应的通道数（每个尺度一个对齐用的 1x1 conv）
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(128, channels, kernel_size=1)
            for channels in channels_list
        ])
        
    def forward(self, features_list, change_map):
        # features_list：多尺度特征（各尺寸不同），change_map：单张变化图
        conditioned_features = []
        for i, features in enumerate(features_list):
            # Down sample change_map to match feature size
            # 先把变化图上采样/下采样到当前特征的分辨率
            _, _, H, W = features.shape
            change_resized = F.interpolate(change_map, size=(H, W), mode='bilinear')
            
            # Compute attention
            # 1x1 conv 把变化图 (1 通道) 对齐到特征通道数 → sigmoid 得注意力图
            attention = torch.sigmoid(self.conv_layers[i](change_resized))
            
            # Condition features
            # 加性门控：attention≈0 时特征原样保留，attention≈1 时特征翻倍（增强）
            conditioned_features.append(features * (1 + attention))
        
        return conditioned_features


class MultiScaleChangeGuidedAttention_StageByStage(nn.Module):
    """逐级变体：输入的多尺度特征与多张变化图逐级一一配对。"""
    def __init__(self, channels_list):
        super().__init__()
        self.activation = torch.sigmoid
        # 每级一个 1x1 conv（128 → 该级通道数）
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(128, channels, kernel_size=1)
            for channels in channels_list
        ])

    def forward(self, feature_maps, change_maps):
        # feature_maps / change_maps：多尺度特征列表与变化图列表（长度一致、逐级配对）
        conditioned_features = []
        for i in range(len(feature_maps)):
            feature_map = feature_maps[i]
            change_map = change_maps[i]
            # Compute attention
            change_map = self.conv_layers[i](change_map)
            attention = self.activation(change_map)

            # print(f"Attention shape: {attention.shape}, Feature map shape: {feature_map.shape}")
            
            # Condition features
            # 同样为加性门控：特征 ×(1+注意力图)
            conditioned_features.append(feature_map * (1 + attention))
        return conditioned_features


class ChangeGuidedAttention(nn.Module):
    """★ 实际使用的 CGA 版本：纯乘法门控，无投影卷积，无状态。

    输入 change_map 与 feature_map 通道数一致（如 512 与 512），
    因此可直接逐元素相乘。核心思想：
        变化响应高的像素 → 语义特征保留；变化响应低的像素 → 语义特征被抑制。
    """
    def __init__(self):
        super().__init__()
        self.activation = torch.sigmoid

    def forward(self, feature_map, change_map):
        # 1) 变化图 → (0,1) 空间注意力图
        attention = self.activation(change_map)
        # Condition features
        # 2) 乘法门控：feature × attention
        conditioned_features = feature_map * attention

        return conditioned_features
