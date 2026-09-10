"""
Mamba-FCS —— 语义变化检测（SCD）总模型装配
============================================
与论文架构图 (docs/full_architecture.png) 对应，数据流如下：

输入：双时相影像
    pre_data  (T1, 前期)      post_data  (T2, 后期)

    ↓ 1. 共享编码器 Backbone_VSSM（只实例化一次，T1/T2 共用一套权重）
         VMamba 骨干对 T1/T2 各编码一次，每期输出 4 级多尺度特征：
         feat_1~feat_4，分辨率约为输入的 1/4、1/8、1/16、1/32，
         通道数为 dims（由配置 EMBED_DIM 翻倍推出，如 128/256/512/1024）

    ↓ 2. 二元变化解码器 ChangeDecoder
         每级做 JSF 空频融合（FFT_Fusion：空间特征 + 对数幅度谱 + |T1-T2|）
         再经 VSSBlock 时空建模、金字塔自上而下融合
         → 二元变化图 output_bcd，并逐级产出 change_maps（4 张变化先验图）

    ↓ 3. 语义解码器 SemanticDecoder × 2（T1、T2 各一个，结构同、参数独立）
         用第 2 步的变化图做 CGA 变化引导：特征 × sigmoid(变化图)
         （变化区域增强、非变化区域抑制）
         → 两期各自的语义分割图 output_T1 / output_T2

    ↓ 4. 分类头 PyramidFusion（main_clf_cd 供 BCD 用，aux_clf 供两期语义共用）
         三个输出统一双线性上采样回原图尺寸，交给训练脚本计算损失

输出：output_bcd      (B,2,H,W)      二元变化图 logits
     output_T1/T2    (B,C,H,W)      两期语义分类图 logits

说明：仅做二元变化检测、不含语义分支的变体见 MambaBCD.STMambaBCD。
"""
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from MambaFCS.changedetection.models.Mamba_backbone import Backbone_VSSM
from MambaFCS.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

# 注：os/time/math/einops/timm/fvcore 等 import 是沿用 VMamba 原仓库的遗留引用，
# 在本文件中并未实际使用；真正用到的核心模块是下面 4 个导入。
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from MambaFCS.changedetection.models.ChangeDecoder import ChangeDecoder
from MambaFCS.changedetection.models.SemanticDecoder import SemanticDecoder
from MambaFCS.changedetection.models.MultiScaleChangeGuidedAttention import MultiScaleChangeGuidedAttention, MultiScaleChangeGuidedAttention_StageByStage
from MambaFCS.changedetection.models.GuidedFusion import PyramidFusion

class STMambaSCD(nn.Module):
    def __init__(self, output_cd, output_clf, pretrained,  **kwargs):
        super(STMambaSCD, self).__init__()
        # ============ 1. 共享编码器：VMamba 骨干 ============
        # 只实例化一次，forward 中对 T1/T2 各跑一遍 → 权重完全共享。
        # out_indices=(0,1,2,3) 取出 4 个 stage 的输出，即 4 级多尺度特征。
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
        # 归一化/激活函数注册表：由 YAML 配置里的字符串
        # （如 'norm_layer: ln2d'）解析成实际 nn 模块，
        # 传给解码器内部的 VSSBlock，使其与骨干保持同一套设置。
        # 注意：channel_first 决定特征是 (B,C,H,W) 还是 (B,H,W,C)，
        # 解码器会据此决定是否插入 Permute（见 ChangeDecoder/SemanticDecoder）。
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)


        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        # ============ 2. 二元变化解码器（BCD）============
        # 吸收 T1/T2 特征 → 输出二元变化图 + 逐级变化先验图 change_maps
        self.decoder_bcd = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        # ============ 3. 语义解码器（SCD）× 2 ============
        # T1、T2 各一个：结构完全相同、参数互不共享。
        # 二者都接收 ChangeDecoder 产出的 change_maps 做 CGA 变化引导。
        self.decoder_T1 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T2 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        # ============ 4. 分类头（PyramidFusion）============
        # main_clf_cd：BCD 输出头（2 类：未变化/变化）
        # aux_clf：两期语义共用的输出头（output_clf 类，即 T1/T2 共享此头部）
        self.main_clf_cd = PyramidFusion(in_channels=128, out_channels=output_cd)
        self.aux_clf = PyramidFusion(in_channels=128, out_channels=output_clf)


    def forward(self, pre_data, post_data):
        # ============ 1. 编码：共享骨干分别编码 T1/T2 ============
        # 每期输出 4 级特征列表（由粗到细）
        # [(B,128,H/4,W/4), (B,256,H/8,W/8), (B,512,H/16,W/16), (B,1024,H/32,W/32)]
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # ============ 2. BCD：二元变化图 + 变化先验图 ============
        # output_bcd：未上采样的二元变化 logit（低分辨率，之后交给分类头）
        # change_maps：4 级变化先验图 [p4, p3, p2, p1]（由粗到细）
        output_bcd, change_maps = self.decoder_bcd(pre_features, post_features)

        change_maps = change_maps[::-1]  # Reverse the order of change maps
        # 反转后为 [p1, p2, p3, p4]（由细到粗），与 SemanticDecoder 内的
        # feat_1~feat_4（由细到粗）一一对应，保证 CGA 时特征与变化图尺度一致。

        # ============ 3. SCD：变化图引导两期语义解码 ============
        # 每个解码器内部用 ChangeGuidedAttention：
        # 特征 × sigmoid(变化图) —— 变化区域增强、非变化区域抑制
        output_T1 = self.decoder_T1(pre_features, change_maps)
        output_T2 = self.decoder_T2(post_features, change_maps)

        # ============ 4. 分类头 + 上采样回原图尺寸 ============
        output_bcd = self.main_clf_cd(output_bcd)
        output_bcd = F.interpolate(output_bcd, size=pre_data.size()[-2:], mode='bilinear')

        output_T1 = self.aux_clf(output_T1)
        output_T1 = F.interpolate(output_T1, size=pre_data.size()[-2:], mode='bilinear')
        
        output_T2 = self.aux_clf(output_T2)
        output_T2 = F.interpolate(output_T2, size=post_data.size()[-2:], mode='bilinear')


        return output_bcd, output_T1, output_T2 
