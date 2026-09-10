"""
train_MambaBCD —— Mamba-FCS 二元变化检测（BCD）训练脚本
========================================================
与 train_MambaSCD 对应，训练"只做二元变化检测"的简化管线：

    模型：STMambaBCD（共享 VMamba 骨干 + ChangeDecoder + 2 类分类头）
    损失：CE + Dice（ce2_dice1）+ 0.5 × Lovász-softmax
    评估：混淆矩阵指标（Recall / Precision / OA / F1 / IoU / Kappa，
          使用 utils_func.metrics.Evaluator 实现）

对应入口：train.py 中把 model_type 配为 MambaBCD_base（视使用方式而定）。
SCD 用户一般无需运行本脚本，仅作 BCD-only 对照实验使用。
"""
import sys
import os
main_dir = os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))
sys.path.append(main_dir)

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
from MambaFCS.changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
from MambaFCS.changedetection.utils_func.metrics import Evaluator
from MambaFCS.changedetection.models.MambaBCD import STMambaBCD
from MambaFCS.changedetection.utils_func.loss import ce2_dice1

import MambaFCS.changedetection.utils_func.lovasz_loss as L

from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # 从 YAML 读取 vssm 模型配置
        config = get_config(args)

        # BCD 训练数据加载器（二元变化标签，无需语义标签）
        self.train_data_loader = make_data_loader(args)

        # 二元评估器（2 类混淆矩阵 → Recall/Precision/OA/F1/IoU/Kappa）
        self.evaluator = Evaluator(num_class=2)

        # ---------- 构建 BCD 模型（骨干参数同 SCD 版本，仅模型骨架不同）----------
        self.deep_model = STMambaBCD(
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
        
        # checkpoint 输出目录
        self.model_save_path = os.path.join(args.model_param_path, f"{args.model_saving_name}")

        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # ---------- 断点续训：加载模型权重 ----------
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        # 优化器：AdamW
        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

        # TensorBoard 日志目录
        self.log_dir = os.path.join(main_dir,'saved_models', f'{args.model_saving_name}')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'logs'))

    def training(self):
        """训练主循环：CE+Dice + 0.5×Lovász，每 500 步验证并按 Kappa 保存最优。"""
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            # 数据解包（BCD 数据集只有 4 元组：pre/post/二值标签/索引）
            pre_change_imgs, post_change_imgs, labels, _ = data

            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            # 前向：只输出一张二元变化图
            output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

            self.optim.zero_grad()
            
            # 主损失：CE(0.5) + Dice(0.5)（见 loss.ce2_dice1）
            ce_loss_1 = ce2_dice1(output_1, labels)

            # 边界精修：Lovász-softmax（直接优化 IoU 代理）
            lovasz_loss = L.lovasz_softmax(F.softmax(output_1, dim=1), labels, ignore=255)
            

            # 总损失：CE+Dice + 0.5 × Lovász
            final_loss = ce_loss_1 + 0.5*lovasz_loss

            final_loss.backward()
            self.optim.step()

            # 记录训练损失曲线
            self.writer.add_scalar('CDLoss/train', final_loss.item(), itera + 1)

            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1}, overall loss is {final_loss}')
                # ---------- 每 500 步验证 + 按 Kappa 保存最优 ----------
                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation()
                    self.writer.add_scalar('CDMetrics/Recall', rec, itera + 1)
                    self.writer.add_scalar('CDMetrics/Precision', pre, itera + 1)
                    self.writer.add_scalar('CDMetrics/OA', oa, itera + 1)
                    self.writer.add_scalar('CDMetrics/F1_score', f1_score, itera + 1)
                    self.writer.add_scalar('CDMetrics/IoU', iou, itera + 1)
                    self.writer.add_scalar('CDMetrics/Kappa', kc, itera + 1)
                    if kc > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model_{kc}.pth'))
                        best_kc = kc
                        best_round = [rec, pre, oa, f1_score, iou, kc]
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)
        self.writer.close()

    def validation(self):
        """在测试集上评估 BCD 指标（混淆矩阵驱动的 Recall/Precision/OA/F1/IoU/Kappa）。"""
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        # 测试数据集（ChangeDetectionDatset：T1/T2 图 + 二值 CD 标签）
        dataset = ChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=16, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels, _ = data
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda()
                labels = labels.cuda().long()

                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

                # argmax → 0/1 预测，逐批喂给混淆矩阵评估器
                output_1 = output_1.data.cpu().numpy()
                output_1 = np.argmax(output_1, axis=1)
                labels = labels.cpu().numpy()

                self.evaluator.add_batch(labels, output_1)
                
        # 汇总输出六项指标
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc
