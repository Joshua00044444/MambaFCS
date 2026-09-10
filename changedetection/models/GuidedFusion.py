"""
GuidedFusion —— 变化检测里的融合模块集合
=========================================
本文件是 Mamba-FCS 的"融合工具箱"，包含 3 组功能：

A. JSF 空频融合（论文核心创新之一）：
   FFTBranch  —— 单张特征 → 2DFFT 对数幅度谱（频域分支）
   FFT_Fusion —— pre/post 空间特征 + 频域幅度谱 + |pre−post| 差分 → 融合图
   ChannelGate / SpatialGate / CrossAttention —— 融合后的通道/空间/双时相精修

B. 金字塔多尺度融合（解码器与分类头复用）：
   PyramidFusion —— 1x1/3x3/5x5 三分支 + CBAM 式通道&空间注意力 + 残差
   ChannelAttention / SpatialAttention —— CBAM 式的通道/空间注意力组件

C. 其他（当前模型未使用，遗留）：
   DepthwiseSeparableConv —— 深度可分离卷积（仅被导入，未实际调用）

使用关系：
   ChangeDecoder/SemanticDecoder 使用 FFT_Fusion 与 PyramidFusion；
   STMambaSCD/MambaBCD 的 main_clf_cd / aux_clf / main_clf 使用 PyramidFusion。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """CBAM 式通道注意力：全局平均池化 + 全局最大池化 → 双全连接 → 相加 → sigmoid。"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享的双层全连接（先缩后升），输出每个通道的权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        # 平均/最大两种池化结果相加 → (B,C,1,1) 通道权重
        return (avg_out + max_out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    """CBAM 式空间注意力：对通道维做均值+最大池化 → 7x7 卷积融合 → sigmoid。"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维压缩成 2 张图：均值图 + 最大图
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        # 7x7 卷积融合 → (B,1,H,W) 空间注意力图
        return self.sigmoid(self.conv(combined))

class PyramidFusion(nn.Module):
    """金字塔多尺度融合（解码器"下采样融合"与分类头的核心模块）。

    三分支并行：1x1（逐像素）、3x3（小感受野）、5x5 深度可分离（大感受野）
    → 拼接 → CBAM 式通道+空间注意力加权 → 1x1 融合 → 残差输出。
    在解码器中承担"粗→细"的通道/分辨率转换（如 1024→512），
    在分类头中承担"128 通道 → 目标类别数"的映射。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 残差快捷分支：通道一致时恒等，否则 1x1 变换
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Multi-scale processing branches
        # 分支1：1x1 逐像素变换
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 分支3：3x3 卷积（小感受野）
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 分支5：5x5 深度可分离（大感受野，group=in_channels 保证轻量）
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # Attention mechanism
        # 注意：channel_att 输入是拼接后的 3*out_channels（CBAM 通道注意力）
        self.channel_att = ChannelAttention(out_channels * 3)
        self.spatial_att = SpatialAttention()
        
        # Fusion
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        
        # Multi-scale features
        # 三个感受野分支并行提取多尺度特征
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        
        # Concatenate and apply attention
        # 拼接 → 通道注意力（重要通道加权）→ 空间注意力（重要位置加权）
        combined = torch.cat([b1, b3, b5], dim=1)
        channel_weights = self.channel_att(combined)
        weighted = combined * channel_weights
        spatial_weights = self.spatial_att(weighted)
        weighted = weighted * spatial_weights
        
        # Final fusion
        # 1x1 融合回 out_channels + 残差
        fused = self.final_conv(weighted)
        return fused + residual

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积（3x3 depthwise + 1x1 pointwise）。遗留模块，未在实际模型中使用。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                 padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))

from einops import rearrange
import math


class CrossAttention(nn.Module):
    """
    A simple Transformer-style cross-attention on flattened feature maps.
    This can capture deeper correlations between two sets of features (pre/post).
    双时相交叉注意力：Q 取 pre、K 和 V 取 post，捕捉两期特征之间的深层关联。
    注意实现细节：Q/K/V 共用一个 self.qkv 线性层，但 forward 中只取
    qkv_q 的第 0 个（Q）与 qkv_kv 的第 0、1 个（K、V）参与计算。
    """
    def __init__(self, dim, num_heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Query, Key, Value projections
        # 共享的 QKV 投影层（对角-切片使用，见 forward 注释）
        self.qkv = nn.Linear(dim, dim * 3, bias=True)

        # Dropouts for attention + final projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        """
        x_q: [B, HW, C] - queries (e.g., pre_feat flattened)
        x_kv: [B, HW, C] - keys/values (e.g., post_feat flattened)
        Returns: [B, HW, C] 
        """
        B, N, C = x_q.shape

        # Project Q from x_q
        # 对 x_q 投影，取第 0 个切片作为 Query（Q）
        qkv_q = self.qkv(x_q).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv_q = qkv_q.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]

        # Project K,V from x_kv
        # 对 x_kv 投影，取第 0、1 个切片作为 Key / Value
        qkv_kv = self.qkv(x_kv).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv_kv = qkv_kv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]

        # Split out Q, K, V
        q = qkv_q[0]  # [B, num_heads, N, head_dim]
        k = qkv_kv[0] # [B, num_heads, N, head_dim]
        v = qkv_kv[1] # [B, num_heads, N, head_dim]
        # (Note: the 3rd slice in each qkv_ array is for Q, but we only need
        #  Q from x_q and K,V from x_kv in this cross-attention design.)

        # Scaled dot-product attention
        # 缩放点积注意力：Q·K^T 缩放 → softmax → 与 V 加权
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_scores = attn_scores.softmax(dim=-1)
        attn_scores = self.attn_drop(attn_scores)

        x_att = attn_scores @ v  # [B, num_heads, N, head_dim]
        x_att = x_att.transpose(1, 2).reshape(B, N, C)  # [B, N, C]

        # Final linear projection
        x_att = self.proj(x_att)
        x_att = self.proj_drop(x_att)

        return x_att



class ChannelGate(nn.Module):
    """
    ECA‑style channel attention.
    Conv1d expects shape [B, 1, C] so we must transpose BEFORE the conv.
    ECA 风格通道注意力：自适应平均池化 → 1D 卷积（核长由通道数自适应）→ sigmoid。
    """
    def __init__(self, C, gamma=2, b=1):
        super().__init__()
        # 自适应 1D 卷积核长 k（ECA 论文公式：k = |(log2(C)+b)/gamma|, 取奇数）
        k = int(abs((math.log2(C) + b) / gamma))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                 # x: [B, C, H, W]
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1)         # [B, C, 1]
        y = self.conv(y.transpose(1, 2))                    # [B, 1, C]（先转置再卷）
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)   # [B, C, 1, 1]
        return x * y


class SpatialGate(nn.Module):
    """门控空间注意力：通道维最大/均值池化 → 7x7 卷积 → sigmoid → 逐像素加权。"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        avg_val = torch.mean(x, dim=1, keepdim=True)
        gate = self.conv(torch.cat([max_val, avg_val], dim=1))
        return x * self.sigmoid(gate)



class FFTBranch(nn.Module):
    """频域分支（JSF 核心）：
    对单张特征图做二维傅里叶变换，取对数幅度谱作为"频域特征"，
    再用 1x1 卷积投影到目标通道数。理由：幅度谱对光照/季节变化更稳健。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1 卷积：把幅度谱投影到 out_channels（保留 H×W 空间结构）
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):                                    # x: [B,C,H,W]
        # 2DFFT（正交归一），取幅度，加 1 后取对数（对数幅度谱）
        # 注意：|fft2| 是实数，.real 仅为形式保留（无实际作用）
        amp = torch.log1p(torch.abs(torch.fft.fft2(x, norm="ortho")))
        return self.proj(amp.real)                           # preserve H×W


class FFT_Fusion(nn.Module):
    """JSF 空频融合（论文核心模块之一，ChangeDecoder 每级各一个）。

    forward 组装：
        cat = [pre_feat, pre_freq, post_feat, post_freq, |pre−post|]
         → 1x1 降维回 in_channels（reduce_conv）
         → ECA 通道门控（ch_gate）→ 空间门控（sp_gate）
    （CrossAttention 已定义但本模块未调用 —— 见下方 forward）
    """
    def __init__(self, in_channels, use_diff=True,
                 cross_attn_heads=4, freq_ratio=1):
        super().__init__()
        self.use_diff = use_diff 
        # 频域分支通道数：in_channels * freq_ratio（默认 1:1 等宽）
        c_freq = int(in_channels * freq_ratio)

        self.FFT_BRANCH = True
        # FFT branches
        # 融合输入通道数 = pre+post 空间（2*C）；若开频域分支则再加上两路频域
        fusion_in = 2* in_channels   
        
        if self.FFT_BRANCH:            # pre+post spatial
            # pre / post 各自一路频域分支
            self.fft_pre  = FFTBranch(in_channels, c_freq)
            self.fft_post = FFTBranch(in_channels, c_freq)

            fusion_in = 2 * (in_channels + c_freq)               # pre+post spatial+freq
        if self.use_diff:
            # 可选：拼接 |pre−post| 差分（时相变化线索）
            fusion_in += in_channels                         # |pre‑post|

        # 1x1 卷积把拼接结果压回 in_channels
        self.reduce_conv = nn.Conv2d(fusion_in, in_channels, 1, bias=False)
        self.reduce_bn   = nn.BatchNorm2d(in_channels)
        self.reduce_relu = nn.ReLU(inplace=True)

        # 融合后的通道/空间门控（ECA 式通道门 + 门控空间注意力）
        self.ch_gate = ChannelGate(in_channels)
        self.sp_gate = SpatialGate()

        # 交叉注意力（定义但未在 forward 中调用 —— 预留）
        self.cross_attn = CrossAttention(dim=in_channels,
                                         num_heads=cross_attn_heads)

    def forward(self, pre_feat, post_feat):
        # 初始拼接：pre 空间 + post 空间
        cat = torch.cat([pre_feat, post_feat], dim=1)

        if self.FFT_BRANCH:
            # 频域分支：pre/post 各自取对数幅度谱，再加进拼接
            pre_freq  = self.fft_pre(pre_feat)
            post_freq = self.fft_post(post_feat)

            cat = torch.cat([pre_feat, pre_freq,
                            post_feat, post_freq], dim=1)

        if self.use_diff:
            # 时相差分特征：|pre_feat − post_feat|（变化位置响应强）
            cat = torch.cat([cat, torch.abs(pre_feat - post_feat)], dim=1)

        # 1x1 融合压缩 → BN → ReLU
        fused = self.reduce_relu(self.reduce_bn(self.reduce_conv(cat)))  # [B,C,H,W]

        # 通道门控 + 空间门控（ECA/门控注意力的轻量精修）
        fused = self.ch_gate(fused)
        fused = self.sp_gate(fused)
        return fused
