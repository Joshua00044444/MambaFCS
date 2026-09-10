"""
utils.py —— 分割/训练通用工具
=================================
【当前被实际使用】
    class2one_hot / simplex / uniq / sset / one_hot
        —— loss.py 的 dice_loss 依赖（one-hot 校验与转换）

【遗留（训练管线未使用）】
    get_scheduler —— 旧式学习率调度（当前训练用 StepLR）
    save_network  —— 旧式检查点保存（当前用 Trainer 内的 torch.save）
"""
import torch
import torch.nn as nn
from torch import Tensor, einsum
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Iterable, Set, Tuple
import logging
import os
logger = logging.getLogger('base')

def simplex(t: Tensor, axis=1) -> bool:
    """校验张量是概率分布：沿 axis 求和是否每行都等于 1（softmax 输出应满足）。"""
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
    """校验是否为 one-hot：即同时满足 simplex 且取值集合 ⊆ {0,1}。"""
    return simplex(t, axis) and sset(t, [0, 1])

def uniq(a: Tensor) -> Set:
    """张量的唯一值集合（如类别 id 集合，CPU 上执行）。"""
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    """判断张量的唯一值集合是否是 sub 的子集（用于校验标签范围）。"""
    return uniq(a).issubset(sub)

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    """标签图 → one-hot 张量（dice_loss 使用）。
    输入：seg [H,W] 或 [B,H,W] 或 [B,1,H,W]；输出 [B,C,H,W] int32。
    断言：标签值 ⊆ [0,C-1]；输出校验为 one-hot。"""
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
    if seg.ndim == 4:
        seg = seg.squeeze(dim=1)
    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    【中文】旧式调度器（按 args['sheduler'] 字典选择 linear/step；当前训练未使用，
    本仓库调度用 train_MambaSCD 中的 StepLR(step_size=10000, gamma=0.5)）。
    """
    if args['sheduler']['lr_policy'] == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args['n_epoch'] + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args['sheduler']['lr_policy'] == 'step':
        step_size = args['n_epoch']//args['sheduler']['n_steps']
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args['sheduler']['gamma'])
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def save_network(opt, epoch, cd_model, optimizer, is_best_model=False ):
    """旧式检查点保存（按 opt['path_cd']['checkpoint'] 目录存 model/opt 两份）。
    【注】当前训练管线未调用本函数（Trainer 直接 torch.save state_dict）。"""
    cd_gen_path = os.path.join(
        opt['path_cd']['checkpoint'], 'cd_model_E{}_gen.pth'.format(epoch))
    cd_opt_path = os.path.join(
        opt['path_cd']['checkpoint'], 'cd_model_E{}_opt.pth'.format(epoch))

    if is_best_model:
        best_cd_gen_path = os.path.join(
            opt['path_cd']['checkpoint'], 'best_cd_model_gen.pth'.format(epoch))
        best_cd_opt_path = os.path.join(
            opt['path_cd']['checkpoint'], 'best_cd_model_opt.pth'.format(epoch))

    # Save CD model pareamters
    network = cd_model
    if isinstance(cd_model, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    # torch.save(state_dict, cd_gen_path)
    if is_best_model:
        torch.save(state_dict, best_cd_gen_path)

    # Save CD optimizer paramers
    opt_state = {'epoch': epoch,
                 'scheduler': None,
                 'optimizer': None}
    opt_state['optimizer'] = optimizer.state_dict()
    # torch.save(opt_state, cd_opt_path)
    if is_best_model:
        torch.save(opt_state, best_cd_opt_path)

    # Print info
    logger.info(
        'Saved current CD model in [{:s}] ...'.format(cd_gen_path))
    if is_best_model:
        logger.info(
            'Saved best CD model in [{:s}] ...'.format(best_cd_gen_path))
