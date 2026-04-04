import os
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import argparse

from utils.utils import setup_seed, get_transform
from models.model import NAGL
from utils.dataset import FSDataset

# import time

from utils.metrics import FewShotMetric

# 单个 epoch 的训练/验证主循环
# - training=True: 前向 + 反向 + 参数更新
# - training=False: 仅前向评估，不更新参数
def run(args, 
        model, 
        dataloader, 
        optimizer=None,  
        training=True):
    
    # 切换模型状态：train 会启用 Dropout/BN 的训练行为，eval 则使用推理行为
    if training:
        model.train()
    else:
        model.eval()
    # 当前 dataloader 中包含的产品类别列表（用于分产品统计指标）
    products = dataloader.dataset.products
    # 记录累计损失，最后计算 epoch 均值
    mean_loss, mean_loss_i, mean_loss_p = 0, 0, 0

    # Few-shot 指标统计器（图像级 AUROC、像素级 AUROC 等）
    fs_metric = FewShotMetric(products)
    # 遍历一个 epoch 内的所有 batch
    for i, data in enumerate(dataloader):
        # 取出 query：query[0] 是图像，query[1] 是像素级 mask
        query = data['query']
        query_image = query[0].to(args.device)  # [B, 1, C, H, W]
        query_mask = query[1].squeeze(1).to(args.device)   # [B, 1, H, W]
        # 当前 batch 对应的产品名（如 bottle、capsule...）
        sample_product = data['sample_product']

        # 图像级标签：0/1（二分类）
        image_level_label = data['image_level_label'][0].to(args.device) # [B]
        
        # 正常/异常 support（由数据集返回，内部含图像与 mask）
        support_normal = data['support_normal'] # (img: [B, n_shot, C, H, W], mask: [B, n_shot, H, W]) or None
        support_abnormal = data['support_abnormal'] # (img: [B, a_shot, C, H, W], mask: [B, a_shot, H, W]) or None
        
        # 前向：得到图像级 logits、像素级 logits，以及两项损失
        image_level_logits, pixel_level_logits, loss_i, loss_p = model(args, query_image, query_mask, image_level_label, support_normal, support_abnormal)
        # 总损失 = 图像级损失 + 像素级损失
        loss = loss_i + loss_p
        
        # 更新指标统计（按类别累计）
        fs_metric.update(image_level_logits, image_level_label, pixel_level_logits, query_mask, sample_product)

        # 累计 loss（用于打印和 epoch 汇总）
        mean_loss += loss
        mean_loss_i += loss_i
        mean_loss_p += loss_p

        # 按 print_freq 打印当前平均损失
        if i % args.print_freq == 0:
            current_iter = i + 1
            print(f'Iter: {i} \t || Total Loss: {mean_loss/current_iter:.4f}, I-Loss: {mean_loss_i/current_iter:.4f}, P-Loss: {mean_loss_p/current_iter:.4f}')

        # 训练模式才进行反向传播与参数更新
        if training:
            # 清空上一轮梯度
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 用优化器更新参数
            optimizer.step()

    # 取 epoch 级别的总体指标
    mean_i_roc, mean_p_roc = fs_metric.get_scores()
    # 打印每类/总体指标
    fs_metric.print_metrics()
    # 返回图像级 AUROC、像素级 AUROC、平均总损失
    return mean_i_roc, mean_p_roc, mean_loss/len(dataloader)

if __name__ == '__main__':
    # 命令行参数解析器
    parser = argparse.ArgumentParser("NAGL", add_help=True)

    # 数据路径配置
    parser.add_argument("--data_root", type=str, default="./path/to/dataset/mvtec", help="train dataset path")
    parser.add_argument("--meta_root", type=str, default="./dataset/meta_json", help="path to meta.json files")
    # parser.add_argument("--data_mode", type=str, default='realiad', choices=['realiad', 'mvtec_visa'], help="train dataset mode")
    parser.add_argument("--data_mode", type=str, default='mvtec_visa', choices=['realiad', 'mvtec_visa'], help="train dataset mode")
    parser.add_argument("--fold", type=int, default=0, help="fold") # 0: val on former (mvtec), training on latter (visa), 1: val on latter (visa), training on former (mvtec)
    parser.add_argument("--save_path", type=str, default='output/checkpoint', help='path to save results')
    
    # dataloader / 分布式配置
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument('--local_rank', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--port', type=str, default='1234', help='number of cpu threads to use during batch generation')
    
    # 模型与 few-shot 配置
    parser.add_argument("--backbone_name", type=str, default='dinov2_vits14', help="the name of encoder")
    parser.add_argument("--dinov2_local_dir", type=str, default="./dinov2", help="local dinov2 repo path, fallback to online when unavailable")
    parser.add_argument("--num_learnable_proxies", type=int, default=3, help="number of learnable queries")
    parser.add_argument("--n_shot", type=int, default=1, help="number of normal samples")
    parser.add_argument("--a_shot", type=int, default=1, help="number of abnormal samples")

    # 训练超参数
    parser.add_argument("--epoch", type=int, default=10, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=512, help="image size")
    parser.add_argument("--print_freq", type=int, default=50, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")

    args = parser.parse_args()

    # 打印完整配置，便于复现实验
    print(args)

    # Set seed
    # 固定随机种子，降低实验波动
    setup_seed(args.seed)
    # 确保输出目录存在
    os.makedirs(args.save_path, exist_ok=True)
    # 若目标 best 文件已存在，则直接跳过（避免重复训练覆盖）
    if os.path.exists(f'{args.save_path}/n_{args.n_shot}_a_{args.a_shot}_best.pth'):
        print(f"Results for N-Shot = {args.n_shot}, A-Shot = {args.a_shot} already exist. Skipping.")
        exit()

    # Distributed setting
    # 从命令行读取当前进程序号（对应 GPU id）
    local_rank = args.local_rank
    # 初始化分布式通信（NCCL: CUDA 上常用后端）
    dist.init_process_group(backend='nccl')
    print('local_rank: ', local_rank)
    # 绑定当前进程到指定 GPU
    torch.cuda.set_device(local_rank)
    # 后续统一使用 args.device
    args.device = torch.device('cuda', local_rank)

    # Create model
    # 构建 NAGL 模型
    model = NAGL(args)
    # Device setup
    # 模型迁移到当前 GPU
    model.to(args.device)
    # 将 BatchNorm 转换为 SyncBatchNorm，支持多卡同步统计量
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # 封装 DDP：多进程分布式训练
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Freeze parameters
    # 这些模块参数被冻结（不参与训练）
    freeze_params_key_name = ['vision_encoder', 
                              'mask_downsample']

    # trainable_params 交给优化器；total/trainable 仅用于打印参数规模
    trainable_params = []
    total_params_num = 0
    trainable_num = 0
    print(f"Trainable Parameters: ")
    # 遍历所有参数，按模块名决定是否训练
    for param_name, param in model.named_parameters():
        if (param_name.split('.')[1] not in freeze_params_key_name):
            print(param_name, param.shape)
            trainable_num += param.numel()
            param.requires_grad_(True)
            trainable_params.append({'params': param})
        else:
            total_params_num += param.numel()
            param.requires_grad_(False)

    # 打印总参数量与可训练参数量（单位：M）
    print(f'Total params: {(total_params_num+trainable_num)/1e6:.3f}M \nTrainable params: {trainable_num/1e6:.3f} M \n')

    # Create optimizer and lr scheduler
    # AdamW 优化器（常用于 Transformer）
    optimizer = torch.optim.AdamW(trainable_params, 
                                  lr=args.learning_rate, 
                                  betas=(0.9, 0.999)
                                  )

    # 多步衰减学习率：在指定 epoch 将 lr 乘以 gamma
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                     milestones=[10, 15], 
                                                     gamma=0.1)
    
    # 构建图像预处理（resize/normalize 等）
    transform = get_transform((args.image_size, args.image_size))

    # Create dataset and dataloader
    # 训练集（episode 采样由 FSDataset 内部完成）
    train_data = FSDataset(data_root=args.data_root,  
                           meta_root=args.meta_root,
                           data_mode=args.data_mode,
                           fold=args.fold, 
                           split='train', 
                           shot=[args.n_shot, args.a_shot], 
                           transform=transform)

    # 分布式采样器：每个进程取数据子集
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    # 训练 dataloader
    train_dataloader = DataLoader(train_data, 
                                  batch_size=args.batch_size, 
                                  pin_memory=True, 
                                  num_workers=args.worker, 
                                  sampler=train_sampler)
    
    # 验证集
    val_data = FSDataset(data_root=args.data_root,
                           meta_root=args.meta_root,
                           data_mode=args.data_mode,
                           fold=args.fold, 
                           split='eval', 
                           shot=[args.n_shot, args.a_shot], 
                           transform=transform)

    # 验证采样器（不打乱）
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    # 验证 dataloader
    val_dataloader = DataLoader(val_data,
                                batch_size=args.batch_size, 
                                pin_memory=True, 
                                num_workers=args.worker, 
                                sampler=val_sampler)

    # Training and validation
    # 记录最佳指标（用 I-AUROC + P-AUROC 作为选择标准）
    best_roc = 0
    # epoch 主循环
    for epoch in range(args.epoch):
        print(f'Epoch: {epoch}, Learning Rate: {scheduler.get_last_lr()[0]}')
        print(f'----------Train-----------')
        # 单个 epoch 训练
        mean_i_roc, mean_p_roc, mean_loss = run(args, 
                        model, 
                        train_dataloader,
                        optimizer, 
                        training=True)
        # 每个 epoch 结束更新学习率
        scheduler.step()
        print(f'Train Results \t || I-AUROC: {mean_i_roc:.4f}, P-AUROC: {mean_p_roc:.4f}, Loss: {mean_loss:.4f}')
        # save the last checkpoint
        # 保存当前 epoch 的“最后权重”（仅保存可训练参数）
        torch.save({name: param for name, param in model.named_parameters() if param.requires_grad}, f'{args.save_path}/n_{args.n_shot}_a_{args.a_shot}_last.pth')

        # 验证阶段不需要梯度，减少显存与计算开销
        with torch.no_grad():
            print(f'--------Validation--------')
            # 单个 epoch 验证
            mean_i_roc, mean_p_roc, mean_loss = run(args, 
                                       model, 
                                       val_dataloader,
                                       training=False)
            print(f'Val Results \t || I-AUROC: {mean_i_roc:.4f}, P-AUROC: {mean_p_roc:.4f}, Loss: {mean_loss:.4f}\n')

            # 预热前 5 个 epoch 不更新 best（降低早期波动影响）
            if (epoch >= 5) and (mean_i_roc + mean_p_roc >= best_roc):
                best_epoch = epoch
                best_i_roc = mean_i_roc
                best_p_roc = mean_p_roc
                best_roc = best_i_roc + best_p_roc
                # save the best model
                # 保存当前最优权重（仅保存可训练参数）
                torch.save({name: param for name, param in model.named_parameters() if param.requires_grad}, f'{args.save_path}/n_{args.n_shot}_a_{args.a_shot}_best.pth')
            
            if epoch < 5:
                print(f'Warmup Epoch {epoch} \n')
            else:
                # 输出历史最优指标与对应 epoch
                print(f'Previous Best I-AUROC: {best_i_roc:.4f}({best_epoch}) || Best P-AUROC: {best_p_roc:.4f}({best_epoch}) \n')
    