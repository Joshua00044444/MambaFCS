"""
STMambaBCD —— 纯二元变化检测（BCD）模型
=========================================
与 STMambaSCD 的对比：Mamba-FCS 的 SCD 版本 = BCD 分支 + 两个语义分支。
本类是"只保留 BCD 部分"的二元变化检测变体：

    STMambaSCD：encoder + ChangeDecoder + SemanticDecoder×2 + 三个分类头
    STMambaBCD：encoder + ChangeDecoder + 一个分类头（无语义分支，无 CGA）

结构（4 级）：
    输入 T1/T2 -->
      共享 Backbone_VSSM 编码（输出 4 级特征）-->
      ChangeDecoder（JSF 空频融合 + VSSBlock 时空建模 + 金字塔融合）
      --> p1 (B,128,H/4,W/4) --> PyramidFusion(128→2) --> 上采样回原尺寸
    输出：二元变化图 logits (B, 2, H, W)

入口剧本：见 changedetection/script/train_MambaBCD.py
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
from MambaFCS.changedetection.models.ChangeDecoder import ChangeDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from MambaFCS.changedetection.models.GuidedFusion import PyramidFusion


class STMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(STMambaBCD, self).__init__()
        # 共享编码器：VMamba 骨干（T1/T2 各跑一次，权重共享），输出 4 级特征
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
        # 归一化/激活函数注册表：与 STMambaSCD 相同的字符串→模块解析方式，
        # 由 YAML 配置决定解码器内 VSSBlock 的归一化/激活设置
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
 

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        # 把已显式传给解码器的 3 个参数从 kwargs 剔除，避免重复传参
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        # BCD 解码器：JSF 空频融合 + 时空建模 + 金字塔融合（输出 p1 与逐级 change_maps，
        # 本类只取 p1 用于二元分类，change_maps 被丢弃）
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        # 二元分类头：PyramidFusion 把最细级 128 通道特征映射为 2 类 logits
        self.main_clf = PyramidFusion(self.encoder.dims[-4], 2)

    def _upsample_add(self, x, y):
        # 与 ChangeDecoder/SemanticDecoder 相同的金字塔上采样加和
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # Encoder processing
        # 1) 共享骨干分别编码 T1/T2，各得到 4 级多尺度特征
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        # 2) BCD 解码器：空频融合 + 时空建模，输出最细级融合特征 p1
        #    （第二个返回值 change_maps 在此被忽略 —— BCD 不需要语义引导）
        output,_ = self.decoder(pre_features, post_features)

        # 3) 分类头 + 上采样回原图尺寸：得到 2 类二元变化图
        output = self.main_clf(output)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        return output
