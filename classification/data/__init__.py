"""data/__init__.py —— 对外导出数据加载器
----------------------------------------------
build_loader(config, simmim=False, is_pretrain=False)：
  simmim=False      → 普通 IMAGENET（build.py）
  simmim=True  且 预训练 → SimMIM 预训练（data_simmim_pt）
  simmim=True  且 微调   → SimMIM 微调（data_simmim_ft）
"""
from .build import build_loader as _build_loader
from .data_simmim_pt import build_loader_simmim
from .data_simmim_ft import build_loader_finetune


def build_loader(config, simmim=False, is_pretrain=False):
    # 普通模式：ImageNet（文件夹/zip）
    if not simmim:
        return _build_loader(config)
    # SimMIM：预训练（掩码图像建模）或微调
    if is_pretrain:
        return build_loader_simmim(config)
    else:
        return build_loader_finetune(config)
