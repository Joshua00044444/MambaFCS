"""
imagenet22k_dataset.py —— ImageNet-22K 数据集（json 标注格式）
==============================================================
【角色】原仓库为追求更高精度，用 22K 数据预训练（21841 类）。
Mamba-FCS 直接用 1K 预训练 vssm_base 权重，本类未参与检测任务。
"""
import os
import json
import torch.utils.data as data
import numpy as np
from PIL import Image

import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class IN22KDATASET(data.Dataset):
    """22K 数据集：按 json 标注（[相对路径, 类id] 列表）加载，读取失败用随机图代替。"""
    def __init__(self, root, ann_file='', transform=None, target_transform=None):
        super(IN22KDATASET, self).__init__()

        self.data_path = root
        self.ann_path = os.path.join(self.data_path, ann_file)
        self.transform = transform
        self.target_transform = target_transform
        # id & label: https://github.com/google-research/big_transfer/issues/7
        # total: 21843; only 21841 class have images: map 21841->9205; 21842->15027
        self.database = json.load(open(self.ann_path))

    def _load_image(self, path):
        """【中文】加载图片（失败时生成 224x224 随机噪点图，保证训练不断档）。"""
        try:
            im = Image.open(path)
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        【中文】返回 (变换后图片, 类索引)。
        """
        idb = self.database[index]

        # images
        images = self._load_image(self.data_path + '/' + idb[0]).convert('RGB')
        if self.transform is not None:
            images = self.transform(images)

        # target
        target = int(idb[1])
        if self.target_transform is not None:
            target = self.target_transform(target)

        return images, target

    def __len__(self):
        return len(self.database)
