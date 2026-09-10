# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

"""
samplers.py —— zip 模式的分片随机采样器（本仓库未使用核心管线）
=================================================================
【角色】原 Swin 仓库：zip+part 缓存模式下，每个 rank 只负责自己分片的
索引子集。Mamba-FCS 检测数据不用 zip，此文件仅保留。
"""
import torch


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    【中文】从给定索引列表做无放回随机采样（用于 zip 分片数据）。
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
