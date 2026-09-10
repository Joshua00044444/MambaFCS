"""
loss.py —— Mamba-FCS 的损失函数集合
=====================================
本文件收集了实验过程中试验过的各种损失，分为三类：

【实际使用（当前训练管线）】
    SeK_Loss          —— ★ 论文核心损失：变化区域内 soft kappa + 频率加权 mIoU
                          （train_MambaSCD 中启用，权重 0.5）
    weighted_BCE_logits —— 类平衡 BCE（被 ce2_dice1 调用）
    ce2_dice1         —— CE + BCE + Dice 组合（train_MambaBCD 使用）
    boundary_loss     —— 边界距离损失（实验性，未在最终管线启用）

【曾用于实验/被注释掉的候选】
    dice_loss / dice_loss_multiclass / ce_dice / ce2_dice1_multiclass
    ce1_dice2 / ce_scl / contrastive_loss / FocalLoss / PerceptualLoss
    tversky_loss / ce2_dice1 的变体、SEK_loss_from_eval（不可微的旧版 SeK）

【训练脚本共用的通用目标函数】：
    class2one_hot / simplex / uniq（来自 utils.py，用于 one-hot 校验与转换）
"""
import os
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, einsum
import torch.nn .functional as F
from typing import Iterable, Set, Tuple
from scipy.ndimage import distance_transform_edt
import torchvision.models as models

from MambaFCS.changedetection.utils_func.utils import simplex, class2one_hot, uniq
from MambaFCS.changedetection.utils_func.mcd_utils import SCDD_eval, SCDD_eval_all

def uniq(a: Tensor) -> Set:
    # 取张量的唯一类别值集合（CPU 上执行）
    return set(torch.unique(a.cpu()).numpy())


def boundary_loss(pred, target):
    """边界距离损失：用距离变换构造边界邻域权重，惩罚靠近边界的错误预测。
    （实验性损失，最终训练管线未使用 —— 现管线用 Lovász 替代边界精修）"""
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()  # [B, C, H, W]

    boundary_distances = []
    for b in range(target.shape[0]):  # Iterate over batch
        for c in range(target_onehot.shape[1]):  # Iterate over classes
            mask = target_onehot[b, c].cpu().numpy()  # Get binary mask for class c
            if np.sum(mask) == 0:  # Skip if no pixels for this class
                boundary_distances.append(np.zeros_like(mask))
                continue
            # Compute the distance transform for the foreground (class c)
            # 正/负类分别做距离变换 → 相加："到边界"的加权距离图
            pos_dist = distance_transform_edt(mask)
            neg_dist = distance_transform_edt(1 - mask)
            boundary_dist = pos_dist + neg_dist
            boundary_distances.append(boundary_dist)
    
    boundary_distances = np.stack(boundary_distances, axis=0)  # [B*C, H, W]
    boundary_distances = torch.from_numpy(boundary_distances).float().to(pred.device)  # Move to GPU if needed

    boundary_distances = boundary_distances.view(pred.shape)  # [B, C, H, W]

    # Compute the boundary loss
    # 边界距离加权后的"软"损失：预测概率 × 距离，越靠近真实边界惩罚越大
    pred_softmax = F.softmax(pred, dim=1)  # Convert logits to probabilities
    loss = torch.mean(pred_softmax * boundary_distances)  # Weighted sum

    return loss


def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    """类平衡的带权 BCE：正类 0.25、负类 0.75 权重，并同时除以类像素数归一化。
    用于 BCD 任务缓解"未变化像素远多于变化像素"的类不平衡。"""
    logit = logit_pixel.reshape(-1)
    truth = truth_pixel.reshape(-1)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    
    pos = (truth>0.5).float()
    neg = (truth<0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos*pos*loss/pos_num + weight_neg*neg*loss/neg_num).sum()

    return loss

def dice_loss(predicts,target,weight=None):
    """Dice 损失（二值场景，前景=变化类）；把目标转成 7 类 on-热后再取 idc=[0,1]。
    （说明：参数 weight 未使用；与 7 类 one-hot 的兼容写法，见 train 中 ce2_dice1）"""
    idc= [0, 1]
    probs = torch.softmax(predicts, dim=1)
    # target = target.unsqueeze(1)
    target = class2one_hot(target, 7)
    assert simplex(probs) and simplex(target)

    pc = probs[:, idc, ...].type(torch.float32)
    tc = target[:, idc, ...].type(torch.float32)
    intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
    union: Tensor = (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

    divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

    loss = divided.mean()
    return loss


def dice_loss_multiclass(pred, target, smooth=1e-6, ignore_index=255):
    """多分类 Dice 损失（可忽略 ignore_index=255 的无标签像素）。"""
    # Mask out invalid pixels
    valid_mask = (target != ignore_index).float()
    target = target.clone()
    target[target == ignore_index] = 0
    
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Apply valid_mask
    pred = pred * valid_mask.unsqueeze(1)
    target_one_hot = target_one_hot * valid_mask.unsqueeze(1)
    
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def ce_dice(input, target, weight=None):
    """CE + Dice 等权组合（通用配方）。"""
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss + 0.5 * dice_loss_
    return loss

def dice(input, target, weight=None):
    """仅 Dice。"""
    dice_loss_ = dice_loss(input, target)
    return dice_loss_

def ce2_dice1(input, target, ignore_index=255):
    """BCD 主配方：CE(×1) + 加权 BCE(×0.35) + Dice(×0.35)。
    其中 BCE 面向"是否变化"的二值逻辑（train_MambaBCD 使用）"""
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    labels_bn = (target > 0).float()  # Binary labels (0 or 1)

    logits_positive = input[:, 1, :, :]  # Shape: [N, H, W]

    bce_loss = weighted_BCE_logits(logits_positive, labels_bn)
    loss = 1 * ce_loss + 0.35 * bce_loss + 0.35* dice_loss_ 
    return loss

def ce2_dice1_multiclass(input, target, weight=None):
    """多分类变体（CE + 多分类 Dice）；当前仅当注释掉的备选，未实际使用。"""
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    target2 = target.clone()
    dice_loss_ = dice_loss_multiclass(input, target2)
    loss = ce_loss #+ 0.25 * dice_loss_ 
    return loss


def ce1_dice2(input, target, weight=None):
    """CE 0.5 + Dice 1.0（Dice 权重更重的组合，实验用）。"""
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss +  dice_loss_
    return loss

def ce_scl(input, target, weight=None):
    """CE + Dice 等权（与 ce_dice 相同，历史遗留别名）。"""
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss + 0.5 * dice_loss_
    return loss

def contrastive_loss(features_1, features_2, label, margin=1.0):
    """无监督/辅助对比损失：未变化区特征"靠近"（相似度→1），变化区特征"远离"。
    （实验性辅助损失，最终管线未使用 —— 现用 similarity 的 MSE 约束）"""
    similarity = F.cosine_similarity(features_1, features_2, dim=1)

    # Loss for unchanged regions (maximize similarity)
    unchanged_loss = (1 - similarity) * (1 - label)  # Mask for unchanged regions

    # Loss for changed regions (minimize similarity)
    changed_loss = torch.clamp(similarity - margin, min=0) * label  # Mask for changed regions

    # Combine losses
    loss = unchanged_loss.mean() + changed_loss.mean()
    return loss

class FocalLoss(nn.Module):
    """Focal Loss：对难样本（概率低）提高权重，缓解类别不平衡。"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=255)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PerceptualLoss(nn.Module):
    """感知损失：用冻结的 VGG16 特征做 MSE。仅在输入是 RGB 图像级时可用。
    （实验性损失，未用于该分割任务）"""
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:16].cuda().eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return F.mse_loss(input_features, target_features)


def tversky_loss(output, target, alpha=0.7, beta=0.3, smooth=1e-6):
    """Tversky 损失：IoU/Dice 的超集（α 控制 FN 惩罚、β 控制 FP 惩罚）。
    （实验性损失，未使用）"""
    # Binary change detection assumed (classes 0: no change, 1: change)
    logits = F.softmax(output, dim=1)
    pred = logits[:, 1, ...]  # Probability of change class
    target = (target == 1).float()  # Convert to binary mask
    
    # Flatten tensors
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Calculate TP, FP, FN
    tp = (pred_flat * target_flat).sum()
    fp = ((1 - target_flat) * pred_flat).sum()
    fn = (target_flat * (1 - pred_flat)).sum()
    
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - tversky


def SEK_loss_from_eval(pred_t1, pred_t2, label_t1, label_t2, change_mask, num_classes):
    """【旧版 SeK】直接用 argmax 后的硬指标算损失（不可微）。
    以 SCDD_eval 的官方 SeK / mIoU 计算 -log 形式损失。训练已改用可微的 SeK_Loss。"""
    label_t1 = label_t1.cuda().long().cpu().numpy()
    label_t2 = label_t2.cuda().long().cpu().numpy()

    pred_t1 = torch.argmax(pred_t1, dim=1).cpu().numpy()
    pred_t2 = torch.argmax(pred_t2, dim=1).cpu().numpy()

    change_mask = torch.argmax(change_mask, axis=1).cpu().numpy()  # Assuming change_mask is one-hot encoded

    # 非变化区域预测强制为 0（与验证后处理一致）
    pred_t1[change_mask == 0] = 0  # Set unchanged pixels to 0
    pred_t2[change_mask == 0] = 0  # Set unchanged pixels to 0

    Fscd_1, IoU_1, SeK_1 = SCDD_eval(pred_t1, label_t1, 37)
    Fscd_2, IoU_2, SeK_2 = SCDD_eval(pred_t2, label_t2, 37)

    average_sek = (SeK_1 + SeK_2) / 2
    average_IoU = (IoU_1 + IoU_2) / 2

    average_sek = torch.tensor(average_sek, dtype=torch.float32).cuda()
    average_IoU = torch.tensor(average_IoU, dtype=torch.float32).cuda()

    sek_loss = -torch.log((average_sek+1)/2 + 1e-6) - 0.5 * torch.log(average_IoU + 1e-6)

    return torch.clamp(sek_loss, min=0.0)


class SeK_Loss(nn.Module):
    """★ SeK 损失（论文核心贡献 3）—— 把评估指标做成可微优化的损失。

    动机：SCD 存在严重的类不均衡（变化像素少、类别长尾），直接用 CE 无法
    命中"评估指标"（Kappa、FWIoU、SeK）。本损失把指标拆成"可微分的形式"：
    在变化区域内，对 T1、T2 的语义预测做：
        1. Kappa 组件：软混淆矩阵 → 观测一致率 po 与期望一致率 pe → Cohen's Kappa；
        2. mIoU 组件：类频加权（1/log(1+freq)）mIoU；
    然后组合为 sek = kappa * exp(β · miou)，再取 -log 形式：
        loss = -log(sek) − γ · log(miou)
    并用 clamp(0) 保证非负。β 控制 IoU 强调程度、γ 控制类别平衡。

    forward 中所有量均在变化像素子集（change_mask==1）上计算 ——
    意思就是只管"改变化了的像素"，不变区域的语义完全不参与 SeK。
    """
    def __init__(self, num_classes, non_change_class=0, beta=1.5, gamma=0.5, eps=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.non_change = non_change_class   # 非变化类 id（计算 Kappa 时排除）
        self.beta = beta  # Controls IoU emphasis
        self.gamma = gamma  # Class balancing
        self.eps = eps
        
    def forward(self, pred_t1, pred_t2, label_t1, label_t2, change_mask):
        """
        Args:
            pred_t1: (B, C, H, W) logits for time 1
            pred_t2: (B, C, H, W) logits for time 2
            label_t1: (B, H, W) ground truth labels T1
            label_t2: (B, H, W) ground truth labels T2
            change_mask: (B, H, W) binary mask (1=changed)
        """
        B, C, H, W = pred_t1.shape
        device = pred_t1.device
        
        # Mask changed regions
        # 只保留变化区域：把变化掩码广播到通道维，softmax 概率乘上掩码
        change_mask = change_mask.unsqueeze(1)  # B,1,H,W
        # 有效类别 = 除"非变化类"外的所有类（Kappa/mIoU 都排除 class 0）
        valid_classes = [c for c in range(C) if c != self.non_change]
        
        # Convert to probabilities
        prob_t1 = F.softmax(pred_t1, dim=1) * change_mask
        prob_t2 = F.softmax(pred_t2, dim=1) * change_mask
        
        # Flatten tensors
        # 全部展平成逐像素向量，便于按变化掩码重新筛选
        prob_t1 = prob_t1.permute(0,2,3,1).reshape(-1, C)  # (N, C)
        prob_t2 = prob_t2.permute(0,2,3,1).reshape(-1, C)
        label_t1 = label_t1.reshape(-1)  # (N)
        label_t2 = label_t2.reshape(-1)
        mask = change_mask.reshape(-1).bool()  # (N)
        
        # Filter changed pixels
        # 只留下变化像素：后续指标只计算变化区域，天然抗"未变化像素淹没"问题
        prob_t1 = prob_t1[mask]
        prob_t2 = prob_t2[mask]
        label_t1 = label_t1[mask]
        label_t2 = label_t2[mask]
        
        if prob_t1.size(0) == 0:  # No changes in batch
            # 整个 batch 没有变化像素：该批贡献 0
            return torch.tensor(0.0).to(device)
            
        # 1. Kappa Component --------------------------------------------------
        def compute_kappa(probs, labels):
            # 软混淆矩阵：概率 × one-hot 标签的外积和（对应"软投票"）
            oh_labels = F.one_hot(labels, C).float()  # (N, C)
            conf_matrix = torch.matmul(probs.T, oh_labels)  # (C, C)
            
            # Exclude non-change class
            # 排除"非变化类"(0)，只看变化类间的混淆
            conf_matrix = conf_matrix[valid_classes][:, valid_classes]
            total = conf_matrix.sum()
            
            # Observed agreement
            po = torch.diag(conf_matrix).sum() / total
            
            # Expected agreement
            row_sum = conf_matrix.sum(dim=1)
            col_sum = conf_matrix.sum(dim=0)
            pe = torch.sum(row_sum * col_sum) / (total ** 2)
            
            # Cohen's Kappa：κ = (po − pe) / (1 − pe)
            return (po - pe) / (1 - pe + self.eps)
            
        kappa_t1 = compute_kappa(prob_t1, label_t1)
        kappa_t2 = compute_kappa(prob_t2, label_t2)
        kappa = (kappa_t1 + kappa_t2) / 2
        
        # 2. mIoU Component ---------------------------------------------------
        def compute_iou(probs, labels):
            """频加权 mIoU：权重 = 1/log(1+freq)，对长尾类别加权更多。"""
            oh_labels = F.one_hot(labels, C).float()
            intersection = (probs * oh_labels).sum(dim=0)[valid_classes]
            union = probs.sum(dim=0)[valid_classes] + oh_labels.sum(dim=0)[valid_classes] - intersection
            
            # Frequency weighting
            freq = oh_labels.sum(dim=0)[valid_classes]
            weights = 1 / torch.log(freq + 1 + self.eps)
            weights = weights / weights.sum()
            
            return (intersection / (union + self.eps) * weights).sum()
            
        iou_t1 = compute_iou(prob_t1, label_t1)
        iou_t2 = compute_iou(prob_t2, label_t2)
        miou = (iou_t1 + iou_t2) / 2
        
        # 3. Combined Loss ----------------------------------------------------
        # SeK 主指标：kappa × exp(β·miou)，再与 mIoU 构成 -log 损失
        sek_value = kappa * torch.exp(self.beta * miou)
        
        log_sek = (sek_value + self.eps).log()
        # log of miou
        self.eps = 1e-6
        log_miou = (miou + self.eps).log()
        # final loss: -log(sek_value) - gamma * log(miou)
        loss = -log_sek - self.gamma * log_miou
        
        return torch.clamp(loss, min=0.0) 
