# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

"""
data_simmim_pt.py —— SimMIM 预训练数据管线（原类别仅支持 swin/swinv2）
=======================================================================
【角色】原 VMamba 仓库用于 SimMIM 预训练（掩码图像建模）；本仓库未使用
（Mamba-FCS 直接使用 VMamba 官方预训练权重，不需要重新预训练）。
核心：MaskGenerator 生成随机掩码 + SimMIMTransform 图像增强。
"""
import math
import random
import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MaskGenerator:
    """SimMIM 掩码生成：把图切成 input_size/mask_patch_size 的格子，
    随机选 mask_ratio 比例格子为"被掩码"，再放大到模型 patch 尺度。
    （本仓库未使用，保留自原仓库）"""
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class SimMIMTransform:
    """SimMIM 变换：图像增强 + 掩码生成，返回 (img, mask)。"""
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        if config.MODEL.TYPE in ['swin', 'swinv2']:
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        else:
            raise NotImplementedError
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask


def collate_fn(batch):
    """【中文】自定义 collate（样本含元组或 None 时逐字段拼接，否则走默认）。"""
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(config):
    """【中文】构建 SimMIM 预训练 DataLoader（返回单元素 dataloader，不含 val）。"""
    transform = SimMIMTransform(config)
    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    return dataloader