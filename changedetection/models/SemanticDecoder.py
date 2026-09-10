"""
SemanticDecoder —— 单一时相的语义分割解码器（SCD 分支）
=========================================================
定位：这是 Mamba-FCS 的语义变化检测（SCD）三分支之一。STMambaSCD 中实例化两个：
decoder_T1 / decoder_T2，分别为 T1、T2 两个时相服务。二者结构完全相同、参数独立。

整体结构与 ChangeDecoder（BCD 分支）同构 —— 同样的金字塔：
    CGA 变化引导 → VSSBlock 时空建模 → ResBlock 平滑 → PyramidFusion 下采样
唯一差异（也是本解码器的灵魂）：每级特征在进入 VSSBlock 之前，
先用 ChangeDecoder 产出的 change_map 做一次"变化引导注意力"。

forward 输入：
    features    : 某一时相的 4 级编码特征，由粗到细
                  (B,1024,H/32,W/32) → (B,128,H/4,W/4)
    change_maps : 4 级变化先验图（注意：由 STMambaSCD 反转后传入，
                  顺序为细→粗，与 features 中的 feat_1~feat_4 按尺度一一对应）
输出：
    p1 : (B, 128, H/4, W/4) 语义 logits（细粒度），
         之后由 STMambaSCD 的 aux_clf（PyramidFusion）升回原图尺寸。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from MambaFCS.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from MambaFCS.changedetection.models.ResBlockSe import ResBlock, SqueezeExcitation
from MambaFCS.changedetection.models.GuidedFusion import PyramidFusion, PyramidFusion, PyramidFusion, FFTBranch
from MambaFCS.changedetection.models.MultiScaleChangeGuidedAttention import ChangeGuidedAttention

import os
main_dir = os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))

class SemanticDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(SemanticDecoder, self).__init__()

        # CGA 总开关：True 时解码器每级都会用变化图条件化语义特征
        self.CHANGE_GUIDED_ATTENTION = True
        # ---------- 4 个时空建模 VSSBlock（由粗到细）----------
        # 命名规则：st_block_N_semantic 的 N 与编码器 stage 对应：
        #   N=4 → coarsest（encoder_dims[-1]=1024, 分辨率 1/32）
        #   N=1 → finest  （encoder_dims[-4]=128,  分辨率 1/4）
        # 每个 block 用 Permute 包裹：channel_first=False（LN 时）特征为
        # (B,H,W,C)，VSSBlock 需要该布局，故外层包 Permute 做转置；
        # channel_first=True（LN2D/BN）时特征本就是 (B,C,H,W)，Permute 退化为恒等。
        # ---------- 细节：与 ChangeDecoder 同为金字塔结构 ----------
        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_4_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=encoder_dims[-1], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_3_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=encoder_dims[-2], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_2_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=encoder_dims[-3], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_1_semantic = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=encoder_dims[-4], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )           

        # ---------- 金字塔下采样（PyramidFusion）----------
        # 每一级把"粗的高维特征"融合下采样到下一级分辨率：
        #   down_sample_1: 1024→512, down_sample_2: 512→256, down_sample_3: 256→128
        # 与 ChangeDecoder 中同名模块作用一致（多尺度卷积分支 + CBAM 注意 + 残差）
        self.down_sample_1 = PyramidFusion(in_channels=encoder_dims[-1], out_channels=encoder_dims[-2])
        self.down_sample_2 = PyramidFusion(in_channels=encoder_dims[-2], out_channels=encoder_dims[-3])
        self.down_sample_3 = PyramidFusion(in_channels=encoder_dims[-3], out_channels=encoder_dims[-4])

        # ---------- 平滑层（ResBlock + SE 注意力）----------
        # 金字塔"自顶向下"融合后、进入 VSSBlock 前，先用残差块平滑低层/高层特征
        # Smooth layer
        self.smooth_layer_3_semantic = ResBlock(in_channels=encoder_dims[-2], out_channels=encoder_dims[-2], stride=1) 
        self.smooth_layer_2_semantic = ResBlock(in_channels=encoder_dims[-3], out_channels=encoder_dims[-3], stride=1)
        self.smooth_layer_1_semantic = ResBlock(in_channels=encoder_dims[-4], out_channels=encoder_dims[-4], stride=1)
        # 最后一层额外在 128 通道上再平滑一次（对应 STMambaSCD 中 aux_clf 的输入通道 128）
        self.smooth_layer_0_semantic = ResBlock(in_channels=128, out_channels=128, stride=1) 
    
    def _upsample_add(self, x, y):
        # 把上一级（更粗）的特征双线性上采样到当前级分辨率，再与当前级原始特征相加
        # （金字塔融合的标准写法："top-down + skip connection"）
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, features, change_maps):
        # features: 单一时间点的 4 级特征（由细到粗：corresponding spatial levels 1/4→1/32）
        # change_maps: 来自 ChangeDecoder 的变化先验图（由细到粗，与 feat_* 尺度一致）
        feat_1, feat_2, feat_3, feat_4 = features
        change_map_1, change_map_2, change_map_3, change_map_4 = change_maps

        # for_figs：收集 CGA 之后的特征（遗留代码，用于画图/可视化，训练中未使用）
        for_figs = []

        '''
            Stage I  —— 最粗级（1/32, 1024 通道）
        '''
        p4 = feat_4
        if self.CHANGE_GUIDED_ATTENTION:
            # ★ 核心：CGA 变化引导 —— 特征 × sigmoid(变化图)
            # 变化图高响应区域保留语义特征，非变化区域特征被抑制
            # 注意：此处每次新建 ChangeGuidedAttention() 实例，但它无参数无状态，功能等价
            p4 = ChangeGuidedAttention()(feat_4, change_map_4)
            for_figs.append(p4)

        # 时空建模（VSSBlock：Mamba 选择性扫描）
        p4 = self.st_block_4_semantic(p4)

        # 下采样融合到下一级（1024→512），p4 作为 stage II 的"粗先验"
        p4 = self.down_sample_1(p4)
        '''
            Stage II  —— 1/16, 512 通道
        '''
        p3 = feat_3
        if self.CHANGE_GUIDED_ATTENTION:
            p3 = ChangeGuidedAttention()(feat_3, change_map_3)
            for_figs.append(p3)
        
        # 上采样加和：p4(粗, 已下采样到 512 通道) + p3(当前级, 已被 CGA 条件化)
        # 注意：此处加的是"被 CGA 条件化过的 p3"，而非 feat_3 原始特征（与 ChangeDecoder 不同）
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3_semantic(p3)
        p3 = self.st_block_3_semantic(p3)

        p3 = self.down_sample_2(p3)
        '''
            Stage III  —— 1/8, 256 通道
        '''
        p2 = feat_2
        if self.CHANGE_GUIDED_ATTENTION:
            p2 = ChangeGuidedAttention()(feat_2, change_map_2)
            for_figs.append(p2)

        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2_semantic(p2)
        p2 = self.st_block_2_semantic(p2)

        p2 = self.down_sample_3(p2)

        '''
            Stage IV  —— 1/4, 128 通道（最细级）
        '''
        p1 = feat_1
        if self.CHANGE_GUIDED_ATTENTION:
            p1 = ChangeGuidedAttention()(feat_1, change_map_1)
            for_figs.append(p1)

        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1_semantic(p1)
        p1 = self.st_block_1_semantic(p1)
        # 最后一层不再下采样（已到最细分辨率），直接 128 通道平滑输出
        p1 = self.smooth_layer_0_semantic(p1) 

        return p1 
