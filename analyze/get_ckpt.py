"""
get_ckpt —— EMA 权重落盘小工具
===============================
训练脚本保存的 checkpoint 里通常同时存在两份权重：
    ckpt["model"]      实时（当前 step）的模型权重
    ckpt["model_ema"]  指数滑动平均（EMA）权重 —— 一般精度更高，评估/发布应使用它

但很多加载代码（--resume、第三方推理脚本）只认 "model" 字段。
本脚本把 EMA 权重拷贝回 "model" 字段，另存为 new_<原文件名>.pth，
从而让任何只读 "model" 的加载路径都能拿到 EMA 权重。

用法（在 analyze/ 目录下执行）：
    python get_ckpt.py
（ckpt 路径写死在 __main__ 的调用里，换文件时改 modema 的参数即可）
"""
import torch
import os


def modema(ckpt=None):
    # realpath 规范化，解析掉符号链接与 "."/".." 段
    src = os.path.realpath(ckpt)
    src_dir = os.path.dirname(src)
    # 输出文件名只用 basename（自动丢弃任何目录成分），固定加 new_ 前缀
    dst = os.path.join(src_dir, "new_" + os.path.basename(src))

    # 路径穿越防护：输入与输出 realpath 的所属目录都必须是 src_dir 本身，
    # 出现 ".." 逃逸出该目录时直接拒绝执行
    for p in (src, dst):
        if os.path.dirname(os.path.realpath(p)) != src_dir:
            raise ValueError(f"路径越界（疑似路径穿越）: {p}")

    # 加载到 CPU（无 GPU 机器也能跑）；weights_only=True 只允许张量/基础类型，
    # 避免 pickle 反序列化不可信 ckpt 时的任意代码执行风险
    _ckpt = torch.load(src, map_location=torch.device("cpu"), weights_only=True)
    _ckpt["model"] = _ckpt["model_ema"]  # 用 EMA 权重覆盖实时权重
    torch.save(_ckpt, dst)

if __name__ == "__main__":
    modema("./vmamba_small_e238_ema.pth")

# Readme: How to use ema ckpts:
# python get_ckpt.py
