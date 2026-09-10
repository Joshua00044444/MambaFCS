# Mamba-FCS 模型架构导读

> 本文档整理自项目代码阅读过程中沉淀的架构说明，配套阅读 `changedetection/models/`
> 目录下已加中文注释的 8 个源码文件。建议顺序：**先读本文档 → 再按第 3 节顺序精读代码**。

---

## 1. 项目一句话概括

**Mamba-FCS（IEEE JSTARS 2026）是一个遥感语义变化检测（SCD）模型**：
输入同一区域的前后两期影像 T1、T2，同时输出
① 二元变化图（哪里变了吗）② 两期语义分类图（变成了什么）。

三大核心创新：

| 创新 | 对应模块 | 作用 |
|---|---|---|
| JSF 空频融合 | `FFT_Fusion` | 把 FFT 对数幅度谱注入空间特征，对光照/季节变化更稳健 |
| CGA 变化引导 | `ChangeGuidedAttention` | 变化图作为空间注意力，引导两期语义解码 |
| SeK 损失 | `SeK_Loss` | 直接优化评估指标（Kappa + 频加权 mIoU） |

底层骨架：**VMamba（Mamba 状态空间模型）骨干**，线性复杂度长距离建模。

---

## 2. 目录地图（models/ 与它的上下游）

```
MambaFCS/
├── changedetection/
│   ├── models/                 ← 本目录（8 个 py 文件，均含中文注释）
│   │   ├── STMambaSCD.py                 总装配（SCD 完整模型）
│   │   ├── Mamba_backbone.py             骨干改造（VMamba → 4 级特征）
│   │   ├── ChangeDecoder.py              BCD 解码器（JSF 金字塔）
│   │   ├── SemanticDecoder.py            SCD 语义解码器（CGA 金字塔）×2
│   │   ├── MambaBCD.py                   BCD-only 变体
│   │   ├── GuidedFusion.py               融合工具箱（FFT_Fusion/PyramidFusion 等）
│   │   ├── MultiScaleChangeGuidedAttention.py  CGA 三种实现（实际用最后一个）
│   │   └── ResBlockSe.py                 ResNet 残差 + SE
│   ├── script/                 train_MambaSCD.py / train_MambaBCD.py
│   ├── datasets/               SECOND / LandSat-SCD 数据加载
│   └── utils_func/            loss.py（SeK_Loss）、mcd_utils.py（SCDD 指标）等
├── classification/models/vmamba.py   VMamba 骨干源码（SS2D/VSSBlock/VSSM）
├── kernels/selective_scan/           选择性扫描 CUDA kernel
└── configs/                          train_LANDSAT.yaml / train_SECOND.yaml
```

---

## 3. 整体数据流（STMambaSCD.forward）

```
pre_data (T1) ──┐                ┌── change_maps["反转后"] ──┐
                 ├→ ① 共享编码器 ─┤                            ↓
post_data (T2) ─┘   Backbone_VSSM │                ② SemanticDecoder(T1) → ③ aux_clf → output_T1
                    (各出 4 级)   ↓                             SemanticDecoder(T2) → ③ aux_clf → output_T2
                   ② ChangeDecoder (BCD)                       ↑
                     │            └──── change_maps ←──────────┘
                     └→ ③ main_clf_cd → output_bcd
```

1. **编码**：共享同一 `Backbone_VSSM`（只实例化一次）分别对 T1/T2 编码，
   各输出 4 级特征 `(128,1/4) (256,1/8) (512,1/16) (1024,1/32)`。
2. **BCD**：`ChangeDecoder` 做 JSF 空频融合 + VSSBlock 时空建模 + 金字塔融合，
   输出二元变化图，同时**逐级保存 change_maps**（变化先验图）。
3. **反转**：`change_maps[::-1]`（由粗到细 → 由细到粗），
   使第 N 级变化图与第 N 级语义特征尺度一一对应。
4. **SCD**：两个 `SemanticDecoder`（T1/T2 各自独立）每级先用 CGA
   （特征 × sigmoid(变化图)）做变化引导，再用同款金字塔做语义分割。
5. **输出**：三个 `PyramidFusion` 分类头（128→2 类、128→C 类）+ 双线性上采样回原图尺寸。

---

## 4. 逐层级调用流程

四段式全景调用链：

```
STMambaSCD.forward(STMambaSCD.py:137)
 ├─ ① 共享编码器  Backbone_VSSM.forward ×2  (Mamba_backbone.py:66)
 ├─ ② BCD 解码器 ChangeDecoder.forward       (ChangeDecoder.py:113)
 ├─ ③ 语义解码器 SemanticDecoder.forward ×2  (SemanticDecoder.py:107)
 └─ ④ 分类头     PyramidFusion.forward ×3    (GuidedFusion.py)
```

### ① 骨干：`Backbone_VSSM.forward`（Mamba_backbone.py:66）

```
x → patch_embed(v2: Conv3x3 s2 ×2 + GELU + LN2d) → (B,128,H/4,W/4)
for stage i in 0..3:
    o = layer.blocks(x)            # 输出收集点（×DEPTHS[i] = 2/2/15/2）
    │   └─ VSSBlock._forward (vmamba.py)
    │        x += drop_path( SS2D( norm(x) ) )      # Mamba 分支
    │        x += drop_path( Mlp( norm2(x) ) )      # FFN 分支
    │        └─ SS2D (forward_type="v3noz")
    │            in_proj(1x1conv) → conv2d(depthwise 3x3) → 激活
    │            → cross_selective_scan (vmamba.py:318) ★核心
    │               ├─ CrossScan: 展成 4 方向序列（水平/垂直 × 正反）
    │               ├─ einsum 生成 dt/B/C 扫描参数
    │               ├─ SelectiveScanOflex.apply（CUDA kernel 前向+反向）
    │               └─ CrossMerge → out_norm(LN2d) → out_proj
    x = layer.downsample(x)        # v3: Conv3x3 s2 + LN（分辨率减半、通道翻倍）
outnorm_i(o) → 收进 outs
```

### ② BCD 解码器：`ChangeDecoder.forward`（ChangeDecoder.py:113）

```
每级流水线（以最粗级 Stage I 为例）：
fuse_layer_1(pre_feat_4, post_feat_4)
 └─ FFT_Fusion.forward (GuidedFusion.py:316)   ← ★ JSF
    1. FFTBranch ×2：log1p(|fft2(pre)|)、log1p(|fft2(post)|)
    2. cat = [pre, pre_freq, post, post_freq, |pre−post|]
    3. reduce_conv(1x1) → BN → ReLU
    4. ch_gate(ECA 通道门控) → sp_gate(7x7 空间门控)
p4 = st_block_41(p4)              # 与骨干同款 VSSBlock（Permute 包裹）
p4_attention = p4                 # ★ 存进 change_maps
p4 = down_sample_1(p4)            # PyramidFusion: 1x1/3x3/5x5 三路 + CBAM + 残差
Stage II~IV：FFT_Fusion → _upsample_add(粗先验) → smooth(ResBlock-SE) → VSSBlock → 下采样
返回：p1(B,128,1/4), change_maps=[p4,p3,p2,p1]
```

### ③ 语义解码器：`SemanticDecoder.forward`（SemanticDecoder.py:107）

与 BCD 金字塔同构，唯一区别是每级最先做 CGA：

```
p4 = ChangeGuidedAttention()(feat_4, change_map_4)   # ★ 实际生效的 CGA
     └─ MultiScaleChangeGuidedAttention.py:59
        attention = sigmoid(change_map)
        return feature_map × attention    # 变化区特征保留、非变化区被抑制
p4 → st_block_4_semantic → down_sample_1 → ... → smooth_layer_0(128通道)
```

### ④ 分类头与输出（STMambaSCD.py:131-134, 144-152）

```python
output_bcd   = interpolate( main_clf_cd(output_bcd),   (H,W) )  # PyramidFusion 128→2
output_T1/T2 = interpolate( aux_clf(output_T1/T2),     (H,W) )  # PyramidFusion 128→C
return output_bcd, output_T1, output_T2   # (B,2,H,W), (B,C,H,W), (B,C,H,W)
```

---

## 5. 关键模块速查

| 模块（文件:类） | 身份与注意点 |
|---|---|
| `STMambaSCD` | 总装配：共享编码器 + 1 个 BCD 解码器 + 2 个语义解码器 + 3 个分类头 |
| `Backbone_VSSM` | 继承 VSSM：删分类头、加 outnorm_i、输出 4 级特征（恒为 channel-first） |
| `ChangeDecoder` | JSF 金字塔，产出 change_maps（BCD→SCD 引导信号的来源） |
| `FFT_Fusion` | ★ JSF：空间 + 频域幅度谱 + 差分；`CrossAttention` 已定义但 forward 未调用 |
| `SemanticDecoder` | CGA 金字塔；`ChangeGuidedAttention()` 每次新建实例（无参数无状态，功能等价） |
| `ChangeGuidedAttention` | ★ 实际生效：`feature × sigmoid(change_map)`（纯乘法门控） |
| `MultiScaleChangeGuidedAttention*` | 两个加性门控变体 `feature × (1+att)`，**未被引用（遗留）** |
| `PyramidFusion` | 三路多尺度卷积 + CBAM 注意 + 残差，兼做下采样融合与分类头 |
| `ResBlock` / `SqueezeExcitation` | 平滑层 + SE 通道注意力 |
| `STMambaBCD` | BCD-only 变体（去掉语义分支与 CGA） |
| `SeK_Loss`（utils_func/loss.py） | ★ 论文损失：变化区域内 soft Kappa + 频加权 mIoU，`-log(sek) − γ·log(miou)` |

---

## 6. 与训练管线的对接

训练入口 `train.py → train_MambaSCD.Trainer`（script/train_MambaSCD.py）：

- **五路损失**：
  `1.0×CE(BCD) + 0.5×(CE(T1)+CE(T2)) + 0.5×Lovász×3 + 0.05×时序一致性MSE + 0.5×SeK`
- **SeK 热启动**：SECOND 从第 0 步启用；Landsat 第 15 万步后启用（权重 0.5）。
- **语义监督范围**：`label_clf[label==0]=255`（ignore）→ 语义损失只监督变化区域。
- **验证**：预测后 `preds[change_mask==0]=0`，再经 `SCDD_eval_all` 输出
  Kappa / Fscd / IoU_mean / SeK / OA，按 SeK 保存最优权重。

---

## 7. 建议阅读顺序（约 2 小时）

1. 本文档（15 min）—— 建立整图
2. `STMambaSCD.py`（20 min）—— 总装配，forward 是对照本文档的"骨架"
3. `GuidedFusion.py` 的 `FFT_Fusion` / `PyramidFusion`（25 min）—— 两个复用最广的模块
4. `ChangeDecoder.py`（20 min）—— JSF 金字塔 + change_maps 产出
5. `SemanticDecoder.py` + `MultiScaleChangeGuidedAttention.py`（20 min）—— CGA 落点
6. `Mamba_backbone.py` + vmamba.py 的 `VSSBlock`/`SS2D`（30 min）—— 找到 `SelectiveScanOflex` 即可
7. `loss.py` 的 `SeK_Loss`（15 min）—— 与 mcd_utils 的 SCDD 指标对照
8. （可选）`MambaBCD.py` / `ResBlockSe.py` —— 5 分钟对照了解

---

## 8. 常见陷阱（读代码时最容易踩）

1. **`change_maps` 顺序**：ChangeDecoder 给出 `[p4,p3,p2,p1]`（粗→细），
   必须反转后给语义解码器（细→粗），CGA 才尺度对齐。
2. **两个 CGA 族**：文件里的 `MultiScale...*` 变体是加性门控 `×(1+att)` 且未被使用；
   `ChangeGuidedAttention` 是纯乘法 `×att`，才是真正生效的。
3. **`channel_first`**：`norm_layer=ln2d` 时解码器 VSSBlock 不需要 Permute；
   默认 ln 时特征为 `(B,H,W,C)`，注意 Permute 包裹。
4. **`FFT_Fusion` 的 CrossAttention**：已定义但 forward 未调用（预留），
   别以为空频融合里包含交叉注意力。
5. **`SemanticDecoder._upsample_add` 加的是 CGA 后的特征**，不是原始 `feat_3`。
6. **`torch.load` 均带 `weights_only=True`**：Mamba_backbone.py 与训练脚本如是，
   读取自定义对象检查点时会走 "Failed loading checkpoint" 分支（可接受）。
