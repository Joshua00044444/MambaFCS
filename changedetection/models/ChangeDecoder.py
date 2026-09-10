"""
ChangeDecoder —— 二元变化检测（BCD）解码器，论文中"空频融合金字塔"的实现
=========================================================================
结构：自上（粗）而下（细）的 4 级金字塔，每一级流水线为：

    FFT_Fusion（JSF 空频融合：空间特征 + 频域对数幅度谱 + 时相差分）
        → VSSBlock（Mamba 时空建模）
        → 下采样融合（PyramidFusion，级联到下一级分辨率时使用）
    （3/2/1 级在加粗先验后先过 ResBlock-SE 平滑，再进 VSSBlock）

与 SemanticDecoder 的关系：
    - 同构金字塔（同款 VSSBlock / PyramidFusion / ResBlock），但这里用的是
      FFT_Fusion（双时相融合），语义解码器用的是单时相特征。
    - 关键产出：每级的融合特征 p4_attention~p1_attention 被收集为 change_maps
      （由粗到细），它们是变化先验，之后被 SemanticDecoder 的 CGA 复用。
      这正是"BCD → SCD 变化引导"中"引导信号"的来源。

forward 输出：
    p1          : (B, 128, H/4, W/4) 最细级融合特征（供 main_clf_cd 分类头）
    change_maps : [p4, p3, p2, p1]   4 级变化先验图（由粗到细）
                  —— STMambaSCD 中会反转顺序后再传给语义解码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from MambaFCS.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from MambaFCS.changedetection.models.ResBlockSe import ResBlock, SqueezeExcitation
from MambaFCS.changedetection.models.GuidedFusion import PyramidFusion, DepthwiseSeparableConv, FFT_Fusion 

class ChangeDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(ChangeDecoder, self).__init__()

        # ---------- 4 级时空建模 VSSBlock（由粗到细）----------
        # 命名规则：st_block_41/31/21/11 的第二个下标 1 无特殊含义（风格性命名），
        # 第一下标对应编码器 stage：4=最粗（encoder_dims[-1]，1/32），1=最细（1/4）。
        # 每级用 Permute 包一层：channel_first=False（LN）时特征为 (B,H,W,C)，
        # 是 VSSBlock 需要的布局；channel_first=True（LN2D/BN）时退化为恒等。
        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_41 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=encoder_dims[-1], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        

        self.st_block_31 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=encoder_dims[-2], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        

        self.st_block_21 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=encoder_dims[-3], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        

        self.st_block_11 = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=encoder_dims[-4], drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'], ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'], ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer, mlp_drop_rate=kwargs['mlp_drop_rate'],
                gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # ---------- JSF 空频融合（FFT_Fusion）4 级 ----------
        # 每级一个：把 pre/post 空间特征 + 各自频域对数幅度谱 + |pre−post| 差分
        # 融合成单张特征图（详见 GuidedFusion.FFT_Fusion）
        self.fuse_layer_1 = FFT_Fusion(in_channels=encoder_dims[-1], use_diff=True)
        self.fuse_layer_2 = FFT_Fusion(in_channels=encoder_dims[-2], use_diff=True)
        self.fuse_layer_3 = FFT_Fusion(in_channels=encoder_dims[-3], use_diff=True)
        self.fuse_layer_4 = FFT_Fusion(in_channels=encoder_dims[-4], use_diff=True)


        # ---------- 金字塔下采样融合（PyramidFusion）----------
        # 把粗级特征（更高维）融合下采样到下一级分辨率：
        #   1024→512 → 512→256 → 256→128
        self.down_sample_1 = PyramidFusion(in_channels=encoder_dims[-1], out_channels=encoder_dims[-2])
        self.down_sample_2 = PyramidFusion(in_channels=encoder_dims[-2], out_channels=encoder_dims[-3])
        self.down_sample_3 = PyramidFusion(in_channels=encoder_dims[-3], out_channels=encoder_dims[-4])
    

        # ---------- 平滑层（ResBlock + SE）----------
        # 金字塔"自顶向下"融合后、进 VSSBlock 前先平滑；最低层（1/4）不再下采样
        # Smooth layer
        self.smooth_layer_3 = ResBlock(in_channels=encoder_dims[-2], out_channels=encoder_dims[-2], stride=1)
        self.smooth_layer_2 = ResBlock(in_channels=encoder_dims[-3], out_channels=encoder_dims[-3], stride=1)
        self.smooth_layer_1 = ResBlock(in_channels=encoder_dims[-4], out_channels=encoder_dims[-4], stride=1)
    

    def _upsample_add(self, x, y):
        # 金字塔融合：把更粗一级的特征双线性上采样到当前级分辨率，再与当前级特征相加
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features):
        change_maps = []
        # 拆包 4 级特征（由粗到细：1/32 → 1/4；注意直接拆包顺序）
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        '''
            Stage I —— 最粗级（1/32，1024 通道）
        '''
        # JSF 空频融合：pre/post 空间 + 频域幅度谱 + 差分 → 单张融合特征图
        p4 = self.fuse_layer_1(pre_feat_4, post_feat_4)
        # 时空建模（VSSBlock：Mamba 选择性扫描）
        p4 = self.st_block_41(p4)

        # ★ 保存本级融合特征 → 变化先验图（供 SemanticDecoder 的 CGA 使用）
        p4_attention = p4
        # 下采样融合到下一级（1024→512），作为 stage II 的粗先验
        p4 = self.down_sample_1(p4)

        '''
            Stage II —— 1/16，512 通道
        '''
        p3 = self.fuse_layer_2(pre_feat_3, post_feat_3)
        # 粗先验上采样加和到本级（p4 已下采样到 512 通道，尺寸即本级分辨率）
        p3 = self._upsample_add(p4, p3)  # Stage number as argument
        p3 = self.smooth_layer_3(p3)
        p3 = self.st_block_31(p3)
        p3_attention = p3

        p3 = self.down_sample_2(p3)

        '''
            Stage III —— 1/8，256 通道
        '''
        p2 = self.fuse_layer_3(pre_feat_2, post_feat_2)
        p2 = self._upsample_add(p3, p2)  # Stage number as argument
        p2 = self.smooth_layer_2(p2)
        p2 = self.st_block_21(p2)
        p2_attention = p2

        p2 = self.down_sample_3(p2)

        '''
            Stage IV —— 1/4，128 通道（最细级）
        '''
        p1 = self.fuse_layer_4(pre_feat_1, post_feat_1)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)
        p1 = self.st_block_11(p1)
        p1_attention = p1

        # 4 级变化先验图（由粗到细）；STMambaSCD 反转后按尺度传给语义解码器
        change_maps = [p4_attention, p3_attention, p2_attention, p1_attention]

        return p1, change_maps
