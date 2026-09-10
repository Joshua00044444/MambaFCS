"""
mcd_utils.py —— 语义变化检测（SCD）评价与常用工具
===================================================
包含三类内容：

【★ SCDD 官方指标】train_MambaSCD 验证的核心依赖：
    SCDD_eval_all   —— 整库 Kappa / Fscd / mIoU / SeK（训练验证使用，37 类统计）
    SCDD_eval       —— 单图版本（仅返回 Fscd / IoU / SeK）
    cal_kappa       —— 从混淆矩阵计算 Cohen's Kappa
    get_hist/fast_hist —— 混淆矩阵累计
    accuracy        —— 逐图像素准确率（Trainer.validation 使用）

【逐图像素指标】
    binary_accuracy / FWIoU / intersectionAndUnion / CaclTP

【数据处理工具】
    read_idtxt / get_square / split_img_into_squares / hwc_to_chw /
    resize_and_crop / batch / seprate_batch / split_train_val /
    normalize / merge_masks / rle_encode / AverageMeter / ImageValStretch2D / ConfMap
"""
import os
import math
import random
import numpy as np
from scipy import stats
from MambaFCS.changedetection.utils_func import eval_segm as seg_acc


def read_idtxt(path):
  """逐字符读一个"无分隔符文本"形式的 id 列表（兼容 0= 与空白分隔）。
  （本项目实际训练用 train.py 的 _read_list_file 逐行读，本函数为历史实现）"""
  id_list = []
  #print('start reading')
  f = open(path, 'r')
  curr_str = ''
  while True:
      ch = f.read(1)
      if is_number(ch):
          curr_str+=ch
      else:
          id_list.append(curr_str)
          #print(curr_str)
          curr_str = ''      
      if not ch:
          #print('end reading')
          break
  f.close()
  return id_list

def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))
    【中文】从拼接图片中取左/右半张（按高度 h 切成左右两个正方形）。"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    """【中文】把 2h×h 的拼接图拆成左右两张（数据探索用）。"""
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    """【中文】HWC → CHW（与 PIL/numpy 通道约定转换）。"""
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=0.5, final_height=None):
    """【中文】按比例缩放后居中裁剪到指定高度（数据预处理遗留工具）。"""
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch
    【中文】把可迭代对象按 batch_size 分批（生成器版本）。"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def seprate_batch(dataset, batch_size):
    """Yields lists by batch
    【中文】按 batch_size 切分 list（一次性返回所有批的旧实现）。"""
    num_batch = len(dataset)//batch_size+1
    batch_len = batch_size
    # print (len(data))
    # print (num_batch)
    batches = []
    for i in range(num_batch):
        batches.append([dataset[j] for j in range(batch_len)])
        # print('current data index: %d' %(i*batch_size+batch_len))
        if (i+2==num_batch): batch_len = len(dataset)-(num_batch-1)*batch_size
    return(batches)

def split_train_val(dataset, val_percent=0.05):
    """【中文】随机按比例划分训练/验证集（旧工具，当前训练直接读 list 文件）。"""
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    """【中文】归一化到 [0,1]（旧工具）。"""
    return x / 255

def merge_masks(img1, img2, full_w):
    """【中文】把左右两张 mask 合并回一张全宽 mask（拼接图拆分的逆操作）。"""
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new

# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    """【中文】RLE 游程编码（掩码压缩存储/Kaggle 格式，辅助工具）。"""
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


class AverageMeter(object):
    """Computes and stores the average and current value
    【中文】平均计：记录 val/count/sum，update 后自动更新 avg（训练验证里统计逐图 OA）。"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def ImageValStretch2D(img):
    """【中文】可视化工具：把 0~1 图乘回 255 并转整型（显示用）。"""
    img = img*255
    #maxval = img.max(axis=0).max(axis=0)
    #minval = img.min(axis=0).min(axis=0)
    #img = (img-minval)*255/(maxval-minval)
    return img.astype(int)

def ConfMap(output, pred):
    """【中文】置信度图：预测类别对应的 softmax 概率 / 各类概率之和（可视化/阈值用）。"""
    # print(output.shape)
    n, h, w = output.shape
    conf = np.zeros(pred.shape, float)
    for h_idx in range(h):
      for w_idx in range(w):
        n_idx = int(pred[h_idx, w_idx])
        sum = 0
        for i in range(n):
          val=output[i, h_idx, w_idx]
          if val>0: sum+=val
        conf[h_idx, w_idx] = output[n_idx, h_idx, w_idx]/sum
        if conf[h_idx, w_idx]<0: conf[h_idx, w_idx]=0
    # print(conf)
    return conf

def accuracy(pred, label, ignore_zero=False):
    """【中文】像素准确率：valid 内 pred==label 的比例。
    ignore_zero=False：label>=0 都算；True：只算 label>0（忽略第0类）。
    返回 (acc, valid_sum) —— train_MambaSCD.validation 逐图统计 OA 时使用。"""
    valid = (label >= 0)
    if ignore_zero: valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum
    
def fast_hist(a, b, n):
    """【中文】把预测(a)与标签(b)的一次性配对累计为 n×n 混淆矩阵（展平写法）。"""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def get_hist(image, label, num_class):
    """【中文】单张图的 n×n 混淆矩阵（行=预测、列=标签）。"""
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist

def cal_kappa(hist):
    """【中文】Cohen's Kappa = (po − pe) / (1 − pe)。
    po=对角占比（观测一致率）；pe=行列边缘概率乘积和（期望一致率）。
    边界处理：全空/pe=1 时返回 0。"""
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa

def SCDD_eval_all(preds, labels, num_class):
    """★ SCDD 基准官方总分（train_MambaSCD.validation 使用的核心函数）。
    输入：多个预测/标签对（全部拼接累计为一个 37 类混淆矩阵），返回：
        kappa_n0 : 把"非变化类(0,0)"置零后算的 Kappa（即只看变化语义一致的 Kappa）
        Fscd     : 变化区域 Precision/Recall 的调和均值（变化检测的 F1）
        IoU_mean : 二值（变化/不变）IoU 平均
        Sek      : 论文优化指标 = kappa_n0 × exp(IoU_fg) / e
    实现细节：37 类混淆矩阵 → 只保留对角线语义一致性：
        c2hist 把"变化/未变化"二值化 → IoU_fg（变化类 IoU）
        hist_n0 把 (0,0) 单元清零 → kappa_n0
    """
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([x for x in range(num_class)])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += get_hist(infer_array, label_array, num_class)
    
    # 二值化（变化/未变化）混淆矩阵：由 (0,0),(0,1),(1,0),(1,1) 组成
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    # 清零"非变化×非变化"单元之后的 Kappa（消除不变类被 0 淹没的影响）
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    # 变化类 IoU + 二值平均 IoU
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    # SeK 主指标（论文定义）
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    
    # 变化区域 Precision / Recall → 调和均值（Fscd）
    pixel_sum = hist.sum()
    change_pred_sum  = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    change_ratio = change_label_sum/pixel_sum
    SC_TP = np.diag(hist[1:, 1:]).sum()
    SC_Precision = SC_TP/change_pred_sum
    SC_Recall = SC_TP/change_label_sum
    Fscd = stats.hmean([SC_Precision, SC_Recall])
    return kappa_n0, Fscd, IoU_mean, Sek

def SCDD_eval(pred, label, num_class):
    """单图版本（SECOND 城市类 = 0~6 时调用，loss.SEK_loss_from_eval 的旧版依赖）。
    只返回 (Fscd, IoU_mean, SeK)。"""
    infer_array = np.array(pred)
    unique_set = set(np.unique(infer_array))
    assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6])), "unrecognized label number"
    label_array = np.array(label)
    assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
    hist = get_hist(infer_array, label_array, num_class)
    # 同样的二值化 + kappa_n0 + IoU + SeK 流程
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
        
    pixel_sum = hist.sum()
    change_pred_sum  = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    change_ratio = change_label_sum/pixel_sum
    SC_TP = np.diag(hist[1:, 1:]).sum()
    SC_Precision = SC_TP/change_pred_sum
    SC_Recall = SC_TP/change_label_sum
    Fscd = stats.hmean([SC_Precision, SC_Recall])
    return Fscd, IoU_mean, Sek

def FWIoU(pred, label, bn_mode=False, ignore_zero=False):
    """【中文】频率加权 IoU（FW-IoU）包装：
    bn_mode=True 时先二值化；ignore_zero=True 时类号减 1（去掉第0类）。"""
    if bn_mode:
        pred = (pred>= 0.5)
        label = (label>= 0.5)
    elif ignore_zero:
        pred = pred-1
        label = label-1
    FWIoU = seg_acc.frequency_weighted_IU(pred, label)
    return FWIoU

def binary_accuracy(pred, label):
    """【中文】二值场景像素准确率（label<2 时统计）。"""
    valid = (label < 2)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc

def intersectionAndUnion(imPred, imLab, numClass):
    """【中文】逐类 intersection / union 统计（直方图法；用 1 作"有效区"占位）。"""
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass+1))
    # print(area_intersection)

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    area_union = area_pred + area_lab - area_intersection
    # print(area_pred)
    # print(area_lab)

    return (area_intersection, area_union)

def CaclTP(imPred, imLab, numClass):
    """【中文】TP / pred / lab 的逐类直方图统计（辅助计算 Precision/Recall/F1）。"""
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # # Remove classes from unlabeled pixels in gt image.
    # # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    TP = imPred * (imPred == imLab)
    (TP_hist, _) = np.histogram(
        TP, bins=numClass, range=(1, numClass+1))
    # print(TP.shape)
    # print(TP_hist)

    # Compute area union:
    (pred_hist, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (lab_hist, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    # print(pred_hist)
    # print(lab_hist)
    # precision = TP_hist / (lab_hist + 1e-10) + 1e-10
    # recall = TP_hist / (pred_hist + 1e-10) + 1e-10
    # # print(precision)
    # # print(recall)
    # F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    # print(F1)


    # print(area_pred)
    # print(area_lab)

    return (TP_hist, pred_hist, lab_hist)
