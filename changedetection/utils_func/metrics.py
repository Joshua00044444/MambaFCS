"""
metrics.py —— 混淆矩阵驱动的分割/变化检测评估器（Evaluator）
==============================================================
基于累计混淆矩阵的一站式评估器（train_MambaBCD.validation 使用）：
    add_batch(gt, pred) 累加混淆矩阵
    Pixel_Accuracy / Pixel_Accuracy_Class / Pixel_Precision_Rate /
    Pixel_Recall_Rate / Pixel_F1_score / Intersection_over_Union /
    Kappa_coefficient / Frequency_Weighted_Intersection_over_Union
    reset() 清零矩阵

混淆矩阵约定：行 = 真实标签、列 = 预测标签（_generate_matrix 中
num_class*gt + pred 的编码式，见 _generate_matrix 注释）。
"""
import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        # 初始化为全零 n×n 混淆矩阵
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.longlong)

    def Pixel_Accuracy(self):
        """像素全局准确率 OA = 对角线之和 / 全部像素和。"""
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        """逐类准确率（平均 mAcc + 每类 Acc 数组）。"""
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-7)
        mAcc = np.nanmean(Acc)
        return mAcc, Acc

    def Pixel_Precision_Rate(self):
        """变化类（class 1）精确率：TP / (FP + TP)。"""
        assert self.confusion_matrix.shape[0] == 2
        Pre = self.confusion_matrix[1, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        return Pre

    def Pixel_Recall_Rate(self):
        """变化类（class 1）召回率：TP / (FN + TP)。"""
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return Rec

    def Pixel_F1_score(self):
        """变化类 F1：2×P×R / (P+R)。"""
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = 2 * Rec * Pre / (Rec + Pre)
        return F1


    def calculate_per_class_metrics(self):
        """按类计算 TP/FN/FP（跳过第 0 类，用于多类别场景；BCD 不常用）。"""
        # Adjustments to exclude class 0 in calculations
        TPs = np.diag(self.confusion_matrix)[1:]  # Start from index 1 to exclude class 0
        FNs = np.sum(self.confusion_matrix, axis=1)[1:] - TPs
        FPs = np.sum(self.confusion_matrix, axis=0)[1:] - TPs
        return TPs, FNs, FPs
    
    def Damage_F1_socore(self):
        """按类计算的 F1 数组（名称沿袭自损坏检测任务，方法通用）。"""
        TPs, FNs, FPs = self.calculate_per_class_metrics()
        precisions = TPs / (TPs + FPs + 1e-7)
        recalls = TPs / (TPs + FNs + 1e-7)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        return f1_scores
    
    def Mean_Intersection_over_Union(self):
        """逐类 IoU 的平均（mIoU）。"""
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix) + 1e-7)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        """变化类（class 1）IoU：TP / (FP + FN + TP)。"""
        IoU = self.confusion_matrix[1, 1] / (
                self.confusion_matrix[0, 1] + self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return IoU

    def Kappa_coefficient(self):
        """Cohen's Kappa：（观测一致率 − 期望一致率）/ (1 − 期望一致率)。
        （下述注释掉的代码为分步写法，功能等价）"""
        # Number of observations (total number of classifications)
        # num_total = np.array(0, dtype=np.long)
        # row_sums = np.array([0, 0], dtype=np.long)
        # col_sums = np.array([0, 0], dtype=np.long)
        # total += np.sum(self.confusion_matrix)
        # # Observed agreement (i.e., sum of diagonal elements)
        # observed_agreement = np.sum(np.diag(self.confusion_matrix))
        # # Compute expected agreement
        # row_sums += np.sum(self.confusion_matrix, axis=0)
        # col_sums += np.sum(self.confusion_matrix, axis=1)
        # expected_agreement = np.sum((row_sums * col_sums) / total)
        num_total = np.sum(self.confusion_matrix)
        observed_accuracy = np.trace(self.confusion_matrix) / num_total
        expected_accuracy = np.sum(
            np.sum(self.confusion_matrix, axis=0) / num_total * np.sum(self.confusion_matrix, axis=1) / num_total)

        # Calculate Cohen's kappa
        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
        return kappa

    def Frequency_Weighted_Intersection_over_Union(self):
        """频率加权 IoU（FW-IoU）：以类别频率为权重的逐类 IoU 加权和。"""
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        """单张图的混淆矩阵生成。
        编码：label = num_class × gt + pred（行=gt、列=pred），
        bincount 统计后 reshape 成 n×n；只统计 0<=gt<n 的像素（掩码过滤）。"""
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """累加一批预测到混淆矩阵（要求两者同形状）。"""
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        """清零混淆矩阵（开始新一轮评估前调用）。"""
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
