"""
train_MambaSCD —— Mamba-FCS 语义变化检测（SCD）训练脚本
========================================================
一键入口（train.py 解析 YAML 后调用本文件的 Trainer）：

    python train.py --config configs/train_LANDSAT.yaml

职责：
    1. 从 YAML 配置构建 STMambaSCD 模型（VMamba 骨干参数全部来自 vssm 配置）；
    2. 训练循环（按迭代次数组织，配合 start_iter 支持断点续训）；
    3. 五路损失联合优化（详见 training() 中 weights 注释）；
    4. 周期性验证，按 SeK 指标保存最优 checkpoint 到 saved_models/<名称>/；
    5. TensorBoard 记录损失与指标曲线。

与 train_MambaBCD 的区别：本脚本训练的是"BCD + 双语义"的 SCD 完整模型，
损失包含语义分类、Lovász、时序一致性（similarity）与 SeK 损失。
"""
import sys
import os
main_dir = os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(__file__))))))

import argparse
import os
import time

import numpy as np

from MambaFCS.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from MambaFCS.changedetection.datasets.make_data_loader import SemanticChangeDetectionDatset, make_data_loader, SemanticChangeDetectionDatset_LandSat
from MambaFCS.changedetection.utils_func.metrics import Evaluator
from MambaFCS.changedetection.models.STMambaSCD import STMambaSCD
import MambaFCS.changedetection.utils_func.lovasz_loss as L
from torch.optim.lr_scheduler import StepLR
from MambaFCS.changedetection.utils_func.mcd_utils import accuracy, SCDD_eval_all, AverageMeter

from MambaFCS.changedetection.utils_func.loss import contrastive_loss, ce2_dice1, ce2_dice1_multiclass, SEK_loss_from_eval, SeK_Loss

from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # 从 YAML 读取 vssm 模型配置（yacs 风格 CfgNode）
        config = get_config(args)

        # 训练数据加载器（SECOND / LandSat 由 args.dataset 决定，内部为两条 data_loader）
        self.train_data_loader = make_data_loader(args)

        # ---------- 构建 SCD 模型 ----------
        # STMambaSCD 需要的骨干参数全部来自配置文件：
        #   dims=EMBED_DIM（VSSM 内部翻倍成 128/256/512/1024）
        #   depths、ssm_*、mlp_*、drop_path_rate 等控制 SS2D/MLP 结构
        # output_cd=2（二元变化图），output_clf=args.num_classes（语义类别数）
        self.deep_model = STMambaSCD(
            output_cd = 2, 
            output_clf = args.num_classes,
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            ) 

        self.deep_model = self.deep_model.cuda()

        # checkpoint 输出目录：model_param_path/模型保存名/
        self.model_save_path = os.path.join(args.model_param_path, f'{args.model_saving_name}')
        self.lr = args.learning_rate
        # 注：epoch 为残值（实际训练按迭代数循环，见 training()）
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # ---------- 断点续训：加载模型权重 ----------
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            # weights_only=True：安全反序列化（仅允许张量/基本类型组成的检查点）
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        # 优化器：AdamW（与官方配置一致：lr=1e-4, weight_decay=5e-4）
        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

        # 学习率调度：每 10000 次迭代减半（StepLR, gamma=0.5）
        self.scheduler = StepLR(self.optim, step_size=10000, gamma=0.5)

        # 续训时同时恢复优化器与调度器状态
        if args.resume is not None:
            self.optim.load_state_dict(torch.load(args.optim_path))
            self.scheduler.load_state_dict(torch.load(args.scheduler_path))

        # TensorBoard 日志目录：saved_models/<名称>/logs
        self.log_dir = os.path.join(main_dir,'saved_models', f'{args.model_saving_name}')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'logs'))

    def training(self):
        """训练主循环：按迭代数遍历数据加载器，5 路损失加权求和后反向传播。"""
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)

        # SeK 损失：只在变化区域内计算 soft kappa + 频率加权 mIoU（论文核心创新之一）
        sek_criterion = SeK_Loss(
            num_classes=self.args.num_classes,  # SECOND dataset classes (exclude non-change)
            non_change_class=0,
            beta=1.5
        ).cuda()

        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            # 数据解包：双时相影像 + 二元变化标签 + 两期语义标签（+ 数据索引）
            pre_change_imgs, post_change_imgs, label_cd, label_clf_t1, label_clf_t2, _ = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            label_cd = label_cd.cuda().long()
            label_clf_t1 = label_clf_t1.cuda().long()
            label_clf_t2 = label_clf_t2.cuda().long()

            # 变化掩码：0=不变区域，1=变化区域（用于 SeK 损失只算变化像素）
            change_mask = (label_cd != 0).float()

            # 语义损失只监督"变化区域"：把非变化区域（label==0）置为 ignore_index(255)
            label_clf_t1[label_clf_t1 == 0] = 255
            label_clf_t2[label_clf_t2 == 0] = 255

            # 前向：输出二元变化图 + 两期语义图
            output_1, output_semantic_t1, output_semantic_t2 = self.deep_model(pre_change_imgs, post_change_imgs)

            pre_change_imgs = pre_change_imgs.float()
            post_change_imgs = post_change_imgs.float()


            # ================== Auxiliary Losses ==================
            # 1. Semantic segmentation losses
            # CE 损失：BCD 图 + 两期语义图（均忽略 255）
            ce_loss_cd = F.cross_entropy(output_1, label_cd, ignore_index=255)
            ce_loss_clf_t1 = F.cross_entropy(output_semantic_t1, label_clf_t1, ignore_index=255)
            ce_loss_clf_t2 = F.cross_entropy(output_semantic_t2, label_clf_t2, ignore_index=255)

            # ce_loss_cd = ce2_dice1(output_1, label_cd)
            # ce_loss_clf_t1 = ce2_dice1_multiclass(output_semantic_t1, label_clf_t1)
            # ce_loss_clf_t2 = ce2_dice1_multiclass(output_semantic_t2, label_clf_t2)

            
            # 2. Boundary refinement
            # Lovász 软损失：直接优化阶梯化 IoU，边界更"锐利"（三路：BCD + T1 + T2）
            lovasz_loss_cd = L.lovasz_softmax(F.softmax(output_1, dim=1), label_cd, ignore=255)
            lovasz_loss_clf_t1 = L.lovasz_softmax(F.softmax(output_semantic_t1, dim=1), label_clf_t1, ignore=255)
            lovasz_loss_clf_t2 = L.lovasz_softmax(F.softmax(output_semantic_t2, dim=1), label_clf_t2, ignore=255)
            
            # 3. Temporal consistency 
            # 时序一致性：在"非变化区域"（label 已被置 255）约束 T1/T2 语义概率一致
            # 即变化区域的语义才允许不同 —— 语义输出与变化检测强绑定
            similarity_mask = (label_clf_t1 == 255).float().unsqueeze(1)
            similarity_loss = F.mse_loss(
                F.softmax(output_semantic_t1, dim=1) * similarity_mask,
                F.softmax(output_semantic_t2, dim=1) * similarity_mask,
                reduction='mean'
            )

            # SeK 损失：变化区域内的 soft 指标优化（见 SeK_Loss 类注释）
            sek_loss_value = sek_criterion(
                output_semantic_t1, 
                output_semantic_t2,
                label_clf_t1,
                label_clf_t2,
                change_mask
            )

            # ================== Loss Weighting ==================
            # 五路损失的权重（默认关闭 SeK；开启后调到 0.5）
            weights = {
                'sek': 0,
                'bcd': 1,
                'ce': 0.5,
                'lovasz': 0.5,
                'similarity': 0.05
            }
            
            # SeK 损失的"热身"：SECOND 从一开始就启用；LandSat 则等 150000 步后再启用
            #（论文经验：指标损失前期不稳定，晚些启用更稳）
            SEK_START_ITER = 0 if self.args.dataset == 'SECOND' else 150000

            if itera + self.args.start_iter > SEK_START_ITER:
                weights['sek'] = 0.5
                weights['bcd'] = 1
                weights['ce'] = 0.5
                weights['lovasz'] = 0.5
                weights['similarity'] = 0.05

            # 总损失 = 五路加权求和
            total_loss = (
                weights['sek'] * sek_loss_value +
                weights['bcd'] * ce_loss_cd +
                weights['ce'] * (ce_loss_clf_t1 + ce_loss_clf_t2) +
                weights['lovasz'] * (lovasz_loss_clf_t1 + lovasz_loss_clf_t2 + lovasz_loss_cd) +
                weights['similarity'] * similarity_loss
            )

            
            # Backpropagation
            # 反向传播 + 更新（每 1 次迭代 = 1 个 batch）
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()
            self.scheduler.step()

            # ---------- 日志 ----------
            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1 + self.args.start_iter}, change detection loss is {weights["bcd"] * ce_loss_cd}, '
                      f'classification loss is {weights["ce"] * (ce_loss_clf_t1 + ce_loss_clf_t2) + weights["lovasz"] * (lovasz_loss_clf_t1 + lovasz_loss_clf_t2)}, '
                      f'SeK loss is {0.5*sek_loss_value}')
                self.writer.add_scalar('Loss/ChangeDetection', weights["bcd"] * ce_loss_cd, itera + 1 + self.args.start_iter)
                self.writer.add_scalar('Loss/Segmentation', 1.4 * sek_loss_value, itera + 1 + self.args.start_iter)
                self.writer.add_scalar('Loss/Classification', weights["ce"] * (ce_loss_clf_t1 + ce_loss_clf_t2) + weights["lovasz"] * (lovasz_loss_clf_t1 + lovasz_loss_clf_t2), itera + 1 + self.args.start_iter)
                self.writer.add_scalar('Loss/Similarity', weights["similarity"] * similarity_loss, itera + 1 + self.args.start_iter)
                self.writer.add_scalar('Loss/Total', total_loss, itera + 1 + self.args.start_iter)
                # ---------- 周期性验证 + 保存最佳 ----------
                # 每 5000 次迭代（或 30000 步后每 1000 次）验证一次，
                # 按 SeK 指标保存最优权重（<iter>_model_<SeK>.pth）
                if ((itera + 1) % 5000 == 0) or (((itera + 1) % 1000 == 0) and ((itera + 1) > 30000)):
                    self.deep_model.eval()
                    kappa_n0, Fscd, IoU_mean, Sek, oa = self.validation()
                    self.writer.add_scalar('Metrics/Kappa', kappa_n0, itera + 1 + self.args.start_iter)
                    self.writer.add_scalar('Metrics/F1', Fscd, itera + 1 + self.args.start_iter)
                    self.writer.add_scalar('Metrics/OA', oa, itera + 1 + self.args.start_iter)
                    self.writer.add_scalar('Metrics/mIoU', IoU_mean, itera + 1 + self.args.start_iter)
                    self.writer.add_scalar('Metrics/SeK', Sek, itera + 1 + self.args.start_iter)
                    if Sek > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                    os.path.join(self.model_save_path, f'{itera + 1 + self.args.start_iter}_model_{Sek:.3f}.pth'))
                        best_kc = Sek
                        best_round = [kappa_n0, Fscd, IoU_mean, Sek, oa ]
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)
        self.writer.close()

    def validation(self):
        """在测试集上评估：输出 Kappa / Fscd / mIoU / SeK / OA。"""
        print('---------starting evaluation-----------')
        dataset = None
        # 数据集选择：SECOND 用 T1/T2/GT 目录布局，LandSat 用 A/B/labelA/labelB 布局
        if self.args.dataset == 'SECOND':
            dataset = SemanticChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')

        if self.args.dataset == 'LandSat':
            dataset = SemanticChangeDetectionDatset_LandSat(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')

        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()
        acc_meter = AverageMeter()

        preds_all = []
        labels_all = []
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_cd, labels_clf_t1, labels_clf_t2, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_cd = labels_cd.cuda().long()
                labels_clf_t1 = labels_clf_t1.cuda().long()
                labels_clf_t2 = labels_clf_t2.cuda().long()


                # input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
                # 前向：得到 BCD 图 + 两期语义图
                output_1, output_semantic_t1, output_semantic_t2 = self.deep_model(pre_change_imgs, post_change_imgs)

                labels_cd = labels_cd.cpu().numpy()
                labels_A = labels_clf_t1.cpu().numpy()
                labels_B = labels_clf_t2.cpu().numpy()

                # 预测的变化掩码：argmax BCD 输出（1=变化）
                change_mask = torch.argmax(output_1, axis=1).cpu().numpy()

                # 预测语义类（argmax），语义只对"变化区域"有效
                preds_A = torch.argmax(output_semantic_t1, dim=1).cpu().numpy()
                preds_B = torch.argmax(output_semantic_t2, dim=1).cpu().numpy()

                # ★ 关键后处理：把非变化区域的语义预测强制置 0（类别 0，非变化）
                #（SCD 评估规定：只有变化区域的语义预测才算数）
                preds_A[change_mask == 0] = 0
                preds_B[change_mask == 0] = 0

                if itera % 100 == 0:
                    print(f'iter is {itera}')

                # 收集逐图准确率（T1/T2 平均）
                for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                    acc_A, valid_sum_A = accuracy(pred_A, label_A)
                    acc_B, valid_sum_B = accuracy(pred_B, label_B)
                    preds_all.append(pred_A)
                    preds_all.append(pred_B)
                    labels_all.append(label_A)
                    labels_all.append(label_B)
                    acc = (acc_A + acc_B) * 0.5
                    acc_meter.update(acc)

        # SCDD 官方指标：Kappa / Fscd / mIoU / SeK 汇总（预测与标签展平后统计）
        kappa_n0, Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, 37)
        print(f'Kappa coefficient rate is {kappa_n0}, F1 is {Fscd}, OA is {acc_meter.avg}, '
              f'mIoU is {IoU_mean}, SeK is {Sek}')
        
        return kappa_n0, Fscd, IoU_mean, Sek, acc_meter.avg
