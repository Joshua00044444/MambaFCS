"""
tp —— 单卡吞吐量（throughput）基准测试
=======================================
衡量模型推理速度：同一批图反复前向，先 50 次预热（触发 CUDA kernel
编译/显存分配），再同步计时 30 次取吞吐 = 30 × batch_size / 耗时(秒)。

数据：ImageNet val 目录，标准评测预处理（短边缩放 256/224 倍例 →
中心裁剪 img_size → ImageNet 均值方差归一化），batch 内 shuffle 关闭。

__main__ 里通过 MODEL 常量切换被测框架（VSSM/SWIN/CONVNEXT/HIVIT/HEAT），
各分支按相对路径动态导入对应仓库的模型类；行尾注释是原作者在
3090/A100 上的历史实测结果，可作对照。
"""
import time
import torch
import torch.utils.data
import argparse
import os
import sys
import logging
from torchvision import datasets, transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.models.vision_transformer import EncoderBlock

# 动态导入工具：临时把目录插到 sys.path，import 后弹出；
# 用于按相对路径引入 VMamba/Swin/ConvNeXt 等外部仓库里的模型类
def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module



# 构建 ImageNet val 的 DataLoader：标准评测预处理
#（短边先缩放到 img_size 的 256/224 倍例，再中心裁剪到 img_size）
def get_dataloader(batch_size=64, root="./val", img_size=224):
    size = int((256 / 224) * img_size)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    dataset = datasets.ImageFolder(root, transform=transform)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    return data_loader


# 吞吐量测试：取一批图，50 次前向预热 → cuda.synchronize 同步 →
# 计时 30 次前向 → 再次同步，throughput = 30 × batch_size / 秒
@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, required=True, help='path to dataset')
    parser.add_argument('--size', type=int, default=224, help='path to dataset')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    dataloader = get_dataloader(
        batch_size=args.batch_size, 
        root=os.path.join(os.path.abspath(args.data_path), "val"),
        img_size=args.size,
    )

    # 通过改 MODEL 常量选择被测框架；各分支从相邻仓库动态导入模型类
    MODEL = "VSSM"

    if MODEL in ["VSSM"]:
        VSSM = import_abspy(
            "vmamba", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../VMamba/classification/models"),
        ).VSSM
        # 不同规格（dims/depths/ssm_ratio/forward_type）的 VSSM 对照，
        # 行尾注释为历史实测吞吐（图/秒）
        model = VSSM(dims=96, depths=[2,2,5,2], ssm_d_state=1, forward_type="v3noz", downsample_version="v3", patchembed_version="v2")
        # INFO:root:batch_size 64 throughput 390.44661250848594
        model = VSSM(dims=96, depths=[2,2,15,2], ssm_d_state=1, forward_type="v3noz", downsample_version="v3", patchembed_version="v2")
        # INFO:root:batch_size 64 throughput 245.8051057770092
        model = VSSM(dims=128, depths=[2,2,15,2], ssm_d_state=1, forward_type="v3noz", downsample_version="v3", patchembed_version="v2")
        # INFO:root:batch_size 64 throughput 175.17029874793926
        model = VSSM(dims=128, depths=[2,2,15,2], ssm_d_state=1, ssm_ratio=1, forward_type="v3noz", downsample_version="v3", patchembed_version="v2")
        # INFO:root:batch_size 64 throughput 383.81260980073216 # A100
        model = VSSM(dims=128, depths=[2,2,15,2], ssm_d_state=1, ssm_ratio=1, forward_type="dev", downsample_version="v3", patchembed_version="v2")
        # INFO:root:batch_size 64 throughput 409.4404580002472 # A100
        
    if MODEL in ["SWIN"]:
        # Swin 用官方仓库实现（可带 CUDA window_shift 算子）
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Swin-Transformer"),)
        # print(sys.path)
        # from kernels.window_process.window_process import WindowProcess, WindowProcessReverse
        SWIN = import_abspy(
            "swin_transformer", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Swin-Transformer/models"),
        ).SwinTransformer
        model = SWIN()
        # INFO:root:batch_size 64 throughput 748.1943721697307
        model = SWIN(embed_dim=96, depths=[2,2,18,2])
        # INFO:root:batch_size 64 throughput 437.82169310230506
        model = SWIN(embed_dim=128, depths=[2,2,18,2], num_heads=[ 4, 8, 16, 32 ])
        # INFO:root:batch_size 64 throughput 284.90212452944024
        # INFO:root:batch_size 64 throughput 410.8976187260478 # A100

    if MODEL in ["CONVNEXT"]:
        CONVNEXT = import_abspy(
            "convnext", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ConvNeXt/models"),
        )
        model = CONVNEXT.convnext_tiny()
        # INFO:root:batch_size 64 throughput 748.796215229091
        model = CONVNEXT.convnext_small()
        # INFO:root:batch_size 64 throughput 437.4729938751549
        model = CONVNEXT.convnext_base()
        # INFO:root:batch_size 64 throughput 284.73416141184316
    
    if MODEL in ["HIVIT"]:
        HIVIT = import_abspy(
            "models_hivit", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../hivit/self_supervised/models"),
        )
        model = HIVIT.hivit_base()
        # INFO:root:batch_size 128 throughput 301.77430420925174
    
    if MODEL in ["HEAT"]:
        HEAT = import_abspy(
            "heat", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"),
        ).HeatM
        model = HEAT(dims=64, depths=[4,4,16,4])
        # INFO:root:batch_size 64 throughput 428.96341939895416
        model = HEAT(dims=112, depths=[4,4,21,4])
        # INFO:root:batch_size 64 throughput 188.57667325700203

    # 建好模型后统一搬到 GPU 并切 eval 模式，再跑吞吐量
    model.cuda().eval()
    throughput(data_loader=dataloader, model=model, logger=logging)

