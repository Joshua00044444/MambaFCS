# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

"""
zipreader.py —— zip 格式数据集读取器
=====================================
把 "xxx.zip@/subdir/file.jpg" 形式的路径解析为 zip 内文件读取；
附带 zip_bank 缓存打开的 zip 文件句柄（避免重复打开）。
Mamba-FCS 检测数据不压缩，此文件仅供原 VMamba 分类预训练使用。
"""
import os
import zipfile
import io
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_zip_path(img_or_path):
    """judge if this is a zip path
    【中文】判断路径是否为 zip 路径（含 '.zip@' 标记）。"""
    return '.zip@' in img_or_path


class ZipReader(object):
    """A class to read zipped files
    【中文】zip 读取器：按 '@' 拆成 zip 路径与内部路径，读取/列举 zip 内文件。"""
    zip_bank = dict()   # zip 路径 → zipfile 句柄缓存

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        """【中文】获取/缓存 zip 句柄。"""
        zip_bank = ZipReader.zip_bank
        if path not in zip_bank:
            zfile = zipfile.ZipFile(path, 'r')
            zip_bank[path] = zfile
        return zip_bank[path]

    @staticmethod
    def split_zip_style_path(path):
        """【中文】按 '@' 拆分：zip 路径（前缀）与内部目录（后缀）。"""
        pos_at = path.index('@')
        assert pos_at != -1, "character '@' is not found from the given path '%s'" % path

        zip_path = path[0: pos_at]
        folder_path = path[pos_at + 1:]
        folder_path = str.strip(folder_path, '/')
        return zip_path, folder_path

    @staticmethod
    def list_folder(path):
        """【中文】列出 zip 内指定目录下的子目录名。"""
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        folder_list = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and \
                    len(os.path.splitext(file_foler_name)[-1]) == 0 and \
                    file_foler_name != folder_path:
                if len(folder_path) == 0:
                    folder_list.append(file_foler_name)
                else:
                    folder_list.append(file_foler_name[len(folder_path) + 1:])

        return folder_list

    @staticmethod
    def list_files(path, extension=None):
        """【中文】列出 zip 内指定目录下匹配扩展名的文件。"""
        if extension is None:
            extension = ['.*']
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        file_lists = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and \
                    str.lower(os.path.splitext(file_foler_name)[-1]) in extension:
                if len(folder_path) == 0:
                    file_lists.append(file_foler_name)
                else:
                    file_lists.append(file_foler_name[len(folder_path) + 1:])

        return file_lists

    @staticmethod
    def read(path):
        """【中文】从 zip 读取文件字节。"""
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        return data

    @staticmethod
    def imread(path):
        """【中文】读取 zip 内图片为 PIL 图（失败时返回随机噪声图）。"""
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        try:
            im = Image.open(io.BytesIO(data))
        except:
            print("ERROR IMG LOADED: ", path_img)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im
