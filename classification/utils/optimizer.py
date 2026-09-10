# --------------------------------------------------------
# Modified by Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

"""
optimizer.py —— 优化器构建（含归一化层不衰减、层衰减等）
==========================================================
build_optimizer：按配置构建 SGD / AdamW，并默认把 1 维参数（norm/bias）
排除在 weight decay 之外；build_optimizer_swimmim（含 layer decay）为
原仓库 microsoft/SimMIM 的遗留代码（本仓库未使用）。
"""
from functools import partial
from torch import optim as optim


def build_optimizer(config, model, logger, **kwargs):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    【中文】构建优化器：bias/norm 不加权重衰减，其余参数按配置衰减。
    """
    logger.info(f"==============> building optimizer {config.TRAIN.OPTIMIZER.NAME}....................")
    skip = {}
    skip_keywords = {}
    # 主干如定义了 no_weight_decay / no_weight_decay_keywords 就纳入跳过名单
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters, no_decay_names = set_weight_decay(model, skip, skip_keywords)
    logger.info(f"No weight decay list: {no_decay_names}")

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    else:
        raise NotImplementedError

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    """【中文】划分 has_decay / no_decay 两组参数：
    一维参数（LayerNorm weight 等）、bias、跳过名单与关键字名单均不衰减。"""
    has_decay = []
    no_decay = []
    no_decay_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_names.append(name)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}], no_decay_names 


def check_keywords_in_name(name, keywords=()):
    """【中文】判断参数名是否命中关键字名单（用于跳过特定层的衰减）。"""
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


# ==========================
# for mim, currently not used, and may have bugs...
# 以下为 SimMIM 预训练/微调遗留的优化器构建（含 layer decay），未使用

def build_optimizer_swimmim(config, model, logger, simmim=True, is_pretrain=False):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    【中文】SimMIM 版：is_pretrain 用 get_pretrain_param_groups，否则按层衰减建组。
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    if is_pretrain:
        parameters = get_pretrain_param_groups(model, skip, skip_keywords)
    else:
        depths = config.MODEL.SWIN.DEPTHS if config.MODEL.TYPE == 'swin' else config.MODEL.SWINV2.DEPTHS
        num_layers = sum(depths)
        get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
        scales = list(config.TRAIN.LAYER_DECAY ** i for i in reversed(range(num_layers + 2)))
        parameters = get_finetune_param_groups(model, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, get_layer_func, scales, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    else:
        raise NotImplementedError

    return optimizer


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    """【中文】SimMIM 预训练参数划分（与 set_weight_decay 相同逻辑，独立实现）。"""
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def get_swin_layer(name, num_layers, depths):
    """【中文】按参数名定位其参与层（Swin 结构），用于微调 layer decay。"""
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    """【中文】微调参数分组：按（层×decay/no_decay）建组，每层学习率 = lr×scale。"""
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())

