Copied from [ChangeMamba Repository](https://github.com/ChenHongruixuan/ChangeMamba).

## 脚本说明（分析工具集）

本目录是论文配套的模型分析脚本（分类骨干层面，与 changedetection 主线代码独立）：

| 脚本 | 用途 |
| --- | --- |
| `tp.py` | 单卡吞吐量基准测试（50 次预热 + 30 次计时前向） |
| `get_flops.py` | FLOPs / 参数量统计（fvcore 与 mmengine 双后端；SelectiveScan 自定义算子已接钩子） |
| `get_erf.py` | 有效感受野（ERF）可视化：特征图中心点对输入像素求梯度，画热力图并统计高贡献区域 |
| `get_scaleup.py` | 分辨率缩放实验：预训练模型在 64~1120 各分辨率下零微调直接验证 + 各分辨率 FLOPs |
| `scaleup_show.py` | 回读 `get_scaleup.py` 的日志，画"精度 vs 分辨率"曲线（import 即执行绘图） |
| `get_loss.py` | 解析三种格式的训练日志（timm / mmpretrain / ConvNeXt），画精度与损失对比曲线 |
| `get_ckpt.py` | 把 checkpoint 里的 EMA 权重拷贝到 "model" 字段，另存 `new_*.pth` |

`mmpretrain_configs/` 为 [mmpretrain](https://github.com/open-mmlab/mmpretrain) 官方配置的拷贝（第三方文件，未改动），供 `classification/models` 的 `build_mmpretrain_models` 构建对比模型（Swin/ConvNeXt/DeiT/ResNet 等）时引用。

各脚本的运行方式与历史实测结果见文件头部的注释。
