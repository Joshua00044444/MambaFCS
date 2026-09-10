"""
Backbone_VSSM —— 面向变化检测改造的 VMamba 骨干
=================================================
直接继承 VMamba 原版 VSSM，做三处改造，使其变成"多尺度特征提取器"：

1. 去掉分类头（del self.classifier），不再输出类别 logits；
2. 为每个 stage 的输出在 append 前做一次 LayerNorm/BatchNorm（outnorm_i）；
3. 通过 out_indices=(0,1,2,3) 收集全部 4 个 stage 的特征并返回
   （每个 stage 取的是 blocks 输出 o —— downsample 之前、当前分辨率下的特征）。

布局约定（重要，下游解码器依赖）：
    VSSM 内部特征布局由 norm_layer 决定：
      - norm_layer='ln' → channel-last (B, H, W, C)
      - norm_layer='ln2d'/'bn' → channel_first (B, C, H, W)
    本类 forward 最后统一转成 (B, C, H, W) 返回，
    因此下游 ChangeDecoder/SemanticDecoder 始终拿到 channel-first 的特征。
"""
from MambaFCS.classification.models.vmamba import VSSM, LayerNorm2d

import torch
import torch.nn as nn


class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', **kwargs):
        # norm_layer='ln'
        # 把 norm_layer 并入 kwargs 传给父类（VSSM 内部据此决定特征布局与归一化方式）
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        # channel_first：特征是否 (B,C,H,W)。'bn' 和 'ln2d' 为 True，'ln' 为 False
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        # 给每个要输出的 stage 挂一个独立归一化层（outnorm0~outnorm3）
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        # 删除父类的分类头（Backbone 只需要特征，不需要分类 logits）
        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        """加载 VMamba 预训练权重（如 vssm_base_0229_ckpt_epoch_237.pth）。
        使用 weights_only=True 安全反序列化（仅允许张量/基本类型组成的检查点）。
        """
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            # strict=False：仅加载键匹配的部分（骨干层），分类头等不匹配的键被忽略
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        # 单个 stage 的前向：先跑 blocks（当前分辨率下的特征变换），
        # 再做 downsample（分辨率/通道减半）作为下一 stage 输入。
        # 注意返回值顺序：o 是 blocks 输出（被收集），x 是 downsample 输出（继续传递）
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)   # (B, C, H/4, W/4)（v2 patch embed 输出 channel_first 取决于 norm_layer）
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C) 或 (B, C, H, W)，取决于 channel_first
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    # 'ln'（channel-last）布局下需要转置成 channel-first 再输出
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        # 返回 4 级特征列表：(B,128,H/4,W/4) → (B,1024,H/32,W/32)（dims 由配置 EMBED_DIM 翻倍推出）
        return outs
