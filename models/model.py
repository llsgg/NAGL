import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange

from models.attention_layer import CrossAttentionLayer, SelfAttentionLayer, PositionEmbeddingSine
from utils.loss import FocalLoss, DiceLoss


class NAGL(nn.Module):
    # 整体模型：
    # 1) 用预训练 DINOv2 提取 query/support 特征
    # 2) 用正常参考得到正常引导分数 s_n
    # 3) 用异常参考 + RM/AFL 模块得到异常引导分数 s_a
    # 4) 融合得到像素级异常分数 a_score，并在训练阶段计算图像级/像素级损失
    def __init__(self, 
                 args
                 ):
        # 先初始化父类 nn.Module 的内部结构（参数注册、子模块容器等）
        super(NAGL, self).__init__()
        # 记录骨干网络名（例如 dinov2_vits14），方便调试/保存配置
        self.backbone_name = args.backbone_name

        # 可选：从本地目录加载 DINOv2（离线环境更稳定）
        local_repo = getattr(args, "dinov2_local_dir", None)
        if local_repo and os.path.isdir(local_repo):
            print(f"Loading DINOv2 from local repo: {local_repo}")
            self.vision_encoder = torch.hub.load(
                local_repo,
                args.backbone_name,
                source='local'
            )
        else:
            if local_repo:
                print(f"Local DINOv2 repo not found at: {local_repo}, fallback to online torch.hub.")
            self.vision_encoder = torch.hub.load(
                'facebookresearch/dinov2',
                args.backbone_name,
            )

        # 特征维度：对 vits14 通常是 384（如果换 backbone，这里要同步检查）
        self.hidden_dim = 384 #vits14
        # patch 尺度：DINOv2 vits14 的 patch 大小为 14
        self.d_scale = 14
        # 骨干在本项目里默认冻结/推理模式（训练时不更新其参数）
        self.vision_encoder.eval()
        # 用卷积把像素级 mask 下采样到 patch 网格尺度
        # 输入输出通道都为 1，本质是“窗口内是否存在异常像素”的聚合
        self.mask_downsample = nn.Conv2d(1, 1, 
                                         kernel_size=self.d_scale, 
                                         stride=self.d_scale, 
                                         padding=1, 
                                         bias=False)
        # 把卷积核全置 1，相当于统计局部窗口中是否有前景像素
        nn.init.constant_(self.mask_downsample.weight, 1.0)

        # 注意力头数
        self.nheads = 4
        # 是否 pre-norm（此处保持 False，对应 attention_layer 实现）
        self.pre_norm = False
        # 可学习 proxy：作为 RM 模块的查询向量（num_proxy x hidden_dim）
        self.learnable_proxies = nn.Embedding(args.num_learnable_proxies, self.hidden_dim)
        # RM：Residual Mining 模块的 cross/self attention
        self.rm_ca, self.rm_sa = self.attention_module() # RM Module
        # AFL：Anomaly Feature Learning 模块的 cross/self attention
        self.afl_ca, self.afl_sa = self.attention_module() # AFL Module
        # 2D 正弦位置编码（给 attention 的 key/value 提供位置信息）
        self.pe_layer = PositionEmbeddingSine(self.hidden_dim//2, normalize=True)

        # 损失函数：图像级 CE + 像素级 Focal + Dice
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

        # ---------- Cross-episode anomaly proxy memory ----------
        # Keep defaults here so old scripts still run.
        self.enable_proxy_memory = getattr(args, "enable_proxy_memory", False)
        self.memory_size = int(getattr(args, "memory_size", 512))
        self.memory_topk = int(getattr(args, "memory_topk", 8))
        self.memory_momentum = float(getattr(args, "memory_momentum", 0.1))
        self.memory_conf_thresh = float(getattr(args, "memory_conf_thresh", 0.6))
        self.memory_dedup_thresh = float(getattr(args, "memory_dedup_thresh", 0.95))
        self.memory_temperature = float(getattr(args, "memory_temperature", 0.07))
        self.memory_alpha = float(getattr(args, "memory_alpha", 0.7))
        self.memory_fuse_mode = getattr(args, "memory_fuse_mode", "dynamic")
        self.memory_warmup_epoch = int(getattr(args, "memory_warmup_epoch", 0))

        self.register_buffer("memory_a", torch.zeros(self.memory_size, self.hidden_dim))
        self.register_buffer("memory_a_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("memory_a_count", torch.zeros(1, dtype=torch.long))

    def attention_module(self):
        # 生成一组 cross-attention + self-attention，供 RM/AFL 复用
        cross_attention = CrossAttentionLayer(                                                                                                                                             
                d_model=self.hidden_dim,
                nhead=self.nheads,
                dropout=0.0,
                normalize_before=self.pre_norm,
            ) 
        self_attention = SelfAttentionLayer(
                d_model=self.hidden_dim,
                nhead=self.nheads,
                dropout=0.0,
                normalize_before=self.pre_norm,
            )
        return cross_attention, self_attention

    @torch.no_grad()
    def feature_forward(self, image):
        # image 形状：(b, num, c, h, w)
        # b: batch 大小；num: 该样本中图片张数（query 常为 1，support 可为 n_shot/a_shot）
        b, num, c, h, w = image.shape
        # 先把 (b, num) 展平，送入 DINOv2 提取 patch 特征
        feat = self.vision_encoder.get_intermediate_layers(image.view(-1, c, h, w))[0]
        # 再还原回 (b, num, patch_num, hidden_dim)
        feat = feat.view(b, num, -1, self.hidden_dim)
        return feat
    
    def get_mask(self, mask, target_size):
        '''
        mask: (b, num, h, w)
        '''
        # 取 batch 维，后续 reshape 需要用到
        b, _, _, _ = mask.shape
        # (b, num, h, w) -> (b*num, 1, h, w)，方便卷积下采样
        mask = rearrange(mask, 'b num h w -> (b num) h w').unsqueeze(1)
        # 下采样到近似 patch 网格尺度
        mask = self.mask_downsample(mask)
        # 只要窗口里出现前景像素，就置为 1（做二值化）
        mask[mask>=1] = 1
        # 如果尺寸仍和目标 patch 网格不一致，做最近邻插值对齐
        if mask.shape[-2] != target_size:
            mask = F.interpolate(mask, size=(target_size, target_size), mode='nearest', align_corners=True)
        # (b*num,1,ph,pw) -> (b,num,ph*pw,1)，与 token 维度对齐
        mask = rearrange(mask, '(b num) 1 p_h p_w -> b num (p_h p_w)', b=b).unsqueeze(-1)
        return mask
    
    def nn_search(self, query_feat, support_feat, support_mask=None, mode='max'):
        '''
        query_feat: (b, num_q, M, c)
        support_feat: (b, num_s, N, c)
        support_mask: (b, num_s, h, w)
        '''
        # M 为 query token 数，N 为 support token 数
        _, num_q, M, _ = query_feat.shape
        _, num_s, N, _ = support_feat.shape
        # 按特征维归一化后再算相似度，等价于余弦相似度更稳定
        query_feat = rearrange(F.normalize(query_feat, dim=-1), 'b num_q M c -> b (num_q M) c')
        support_feat = rearrange(F.normalize(support_feat, dim=-1), 'b num_s N c -> b (num_s N) c')
        # query_feat = rearrange(query_feat, 'b 1 M c -> b M c')
        # support_feat = rearrange(support_feat, 'b num_s N c -> b (num_s N) c')
        if support_mask is not None:
            # 像素级 mask -> token 级 mask
            support_mask_ = self.get_mask(support_mask, int(N**0.5))
            # 扩展到与 similarity_map 形状兼容：(b, M, num_s*N)
            support_mask = rearrange(support_mask_, 'b num_s N 1 -> b 1 (num_s N)').repeat(1, M, 1)
        else:
            # 无 mask 时，相当于所有 support token 都可参与匹配
            support_mask_ = None
            support_mask = 1

        # 计算 query 与 support 的两两相似度：
        # einsum 后为 (b, num_q*M, num_s*N)，再映射到 [0,1] 并应用 mask
        similarity_map = (1+torch.einsum('bmc,bnc->bmn', query_feat, support_feat))/2*support_mask
        if mode == 'max':
            # 对每个 query token 取“最相似”的 support token 作为匹配分数
            pseudo_mask = similarity_map.max(dim=-1)[0]
        elif mode == 'mean':
            # 对 support token 求均值，得到更平滑的匹配分数
            pseudo_mask = similarity_map.mean(dim=-1)
        # 还原为 (b, num_q, M, 1)
        pseudo_mask = rearrange(pseudo_mask, 'b (num_q M) -> b num_q M 1', num_q=num_q)

        # 返回 query 的伪分数图 + 下采样后的 support mask（供后续模块复用）
        return pseudo_mask, support_mask_

    def attention_forward(self, cross_layer, self_layer, query_embed, key_feat, value_feat, feat_mask=None):
        '''
        query_embed: (num_q, c)
        key_feat: (b, num_s, M, c)
        value_feat: (b, num_s, M, c)
        feat_mask: (b, num_s, M, 1)
        '''
        # B 为 batch，C 为通道维
        B, _, _, C = value_feat.shape

        if isinstance(query_embed, nn.Embedding):
            # gaussian init
            # nn.init.normal_(query_embed.weight, mean=0, std=0.02)
            # learnable embedding 形状 (num_q,c) -> (num_q,B,c)
            q_supp_out = query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        elif query_embed.dim() == 2:
            # 直接给定 (num_q,c) 时，同样扩展到 batch 维
            q_supp_out = query_embed.unsqueeze(1).repeat(1, B, 1)
        else:
            # 若输入已是 (B,num_q,c)，转成 (num_q,B,c) 以匹配 MultiheadAttention
            q_supp_out = query_embed.permute(1, 0, 2)

        # key/value 展平成 sequence-first 格式：(seq_len, B, C)
        key = rearrange(key_feat, 'b num M c -> (num M) b c')
        # 根据 value 和 mask 生成 2D 位置编码
        pos_embedding = self.pe_layer(value_feat, feat_mask)
        value = rearrange(value_feat, 'b num M c -> (num M) b c')

        if feat_mask is not None:
            # 构建 attention mask：允许位置为 1，屏蔽位置为 -1e9
            # 形状需匹配 (num_heads * B, target_len, source_len)
            attn_mask = rearrange(feat_mask.squeeze(-1), 'b num M -> b (num M)').unsqueeze(1).repeat(self.nheads, q_supp_out.shape[0], 1)
            attn_mask = -1e9*(1-attn_mask)
        else:
            attn_mask = feat_mask

        # 先 cross-attention：query 从 key/value 中读取信息
        output = cross_layer(q_supp_out, key, value, 
                             memory_mask=attn_mask, 
                             memory_key_padding_mask=None,
                             pos=pos_embedding, query_pos=None
                             )
        # 再 self-attention：让 query token 之间交互
        output = self_layer(output, 
                            tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                            )

        # (num_q,B,C) -> (B,num_q,C)，便于后续模块使用
        return output.permute(1, 0, 2)

    def get_res_feat(self, ori_feat, temp_feat):
        '''
        ori_feat: (b, num, h*w, c)
        temp_feat: (b, num, h*w, c)
        '''
        # 拆出维度信息
        b, num, h_w, c = ori_feat.shape
        # 展平成 token 序列，便于两两相似度计算
        ori_feat = rearrange(ori_feat, 'b num h_w c -> b (num h_w) c')
        temp_feat = rearrange(temp_feat, 'b num h_w c -> b (num h_w) c')
        # 计算每个 ori token 与 temp token 的相似度
        sim_map = (1+torch.einsum('bmc,bnc->bmn', ori_feat, temp_feat))/2
        # 对每个 ori token，找到最相似的 temp token 索引
        max_idx = sim_map.max(dim=-1)[1]
        # gather 取出对应的最相似特征向量
        most_sim_feat = torch.gather(temp_feat, 1, max_idx.unsqueeze(-1).repeat(1, 1, c))
        # 残差特征 = 原特征 - 最相似模板特征
        res_feat = ori_feat - most_sim_feat
        # 还原回 (b,num,h*w,c)
        res_feat = rearrange(res_feat, 'b (num h_w) c -> b num h_w c', b=b, num=num)
        return res_feat

    def proxy_confidence(self, proxy_tokens, ref_feat):
        """
        proxy_tokens: (b, p, c)
        ref_feat: (b, num, m, c)
        return: (b, p) in [0, 1]
        """
        ref_feat = rearrange(F.normalize(ref_feat, dim=-1), 'b num m c -> b (num m) c')
        proxy_tokens = F.normalize(proxy_tokens, dim=-1)
        sim = torch.einsum('bpc,brc->bpr', proxy_tokens, ref_feat)
        conf = (1 + sim.max(dim=-1)[0]) / 2
        return conf.clamp(0.0, 1.0)

    def retrieve_anomaly_memory(self, proxy_tokens):
        """
        proxy_tokens: (b, p, c)
        returns:
            mem_proxy: (b, p, c)
            mem_conf: (b, p) in [0,1]
        """
        valid_count = int(self.memory_a_count.item())
        if (not self.enable_proxy_memory) or valid_count == 0:
            mem_proxy = torch.zeros_like(proxy_tokens)
            mem_conf = torch.zeros(proxy_tokens.shape[:2], device=proxy_tokens.device, dtype=proxy_tokens.dtype)
            return mem_proxy, mem_conf, False

        memory = self.memory_a[:valid_count]
        proxy_norm = F.normalize(proxy_tokens, dim=-1)
        memory_norm = F.normalize(memory, dim=-1)
        sim = torch.einsum('bpc,kc->bpk', proxy_norm, memory_norm)

        topk = min(self.memory_topk, valid_count)
        top_scores, top_indices = torch.topk(sim, k=topk, dim=-1)
        weights = F.softmax(top_scores / self.memory_temperature, dim=-1)

        selected_memory = memory[top_indices]  # (b, p, topk, c)
        mem_proxy = (weights.unsqueeze(-1) * selected_memory).sum(dim=-2)
        mem_conf = ((1 + top_scores[..., 0]) / 2).clamp(0.0, 1.0)
        return mem_proxy, mem_conf, True

    def fuse_anomaly_proxies(self, current_proxy, memory_proxy, current_conf, memory_conf):
        """
        current_proxy/memory_proxy: (b, p, c)
        current_conf/memory_conf: (b, p)
        """
        if self.memory_fuse_mode == "fixed":
            alpha = torch.full_like(current_conf, self.memory_alpha)
        else:
            alpha = current_conf / (current_conf + memory_conf + 1e-6)
            alpha = alpha.clamp(0.1, 0.9)
        fused_proxy = alpha.unsqueeze(-1) * current_proxy + (1 - alpha).unsqueeze(-1) * memory_proxy
        return fused_proxy, alpha

    @torch.no_grad()
    def update_anomaly_memory(self, proxy_tokens, conf_scores, current_epoch=0):
        """
        proxy_tokens: (b, p, c)
        conf_scores: (b, p)
        """
        if not self.enable_proxy_memory:
            return
        if current_epoch < self.memory_warmup_epoch:
            return

        token_list = proxy_tokens.reshape(-1, proxy_tokens.shape[-1]).detach()
        conf_list = conf_scores.reshape(-1).detach()
        valid_mask = conf_list > self.memory_conf_thresh
        if valid_mask.sum() == 0:
            return

        token_list = F.normalize(token_list[valid_mask], dim=-1)
        memory = self.memory_a
        K = memory.shape[0]
        ptr = int(self.memory_a_ptr.item())
        count = int(self.memory_a_count.item())

        for token in token_list:
            if count > 0:
                existing = F.normalize(memory[:count], dim=-1)
                max_sim = torch.matmul(existing, token).max()
                if max_sim > self.memory_dedup_thresh:
                    continue

            if count < K:
                memory[count] = token
                count += 1
                ptr = count % K
            else:
                memory[ptr] = (1 - self.memory_momentum) * memory[ptr] + self.memory_momentum * token
                ptr = (ptr + 1) % K

        self.memory_a_ptr[0] = ptr
        self.memory_a_count[0] = count
    
    def prepare_test_image(self, img, transform):
        # 推理时允许传路径字符串
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        # 也允许传 numpy 数组
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        # 应用预处理（resize/normalize/tensor 等）
        image_tensor = transform(img)
        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        # 保证高宽是 patch_size 的整数倍，避免 ViT patch 切分错位
        cropped_width, cropped_height = width - width % self.vision_encoder.patch_size, height - height % self.vision_encoder.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        # 返回 patch 网格大小，后处理可用
        grid_size = (cropped_height // self.vision_encoder.patch_size, cropped_width // self.vision_encoder.patch_size)
        # return image_tensor
        return image_tensor, grid_size
    
    def forward(self, args, query_image, query_mask, query_label, support_normal, support_abnormal, mode='train'):

        # query 图像提取 token 特征：(b,1,M,c)
        query_feat = self.feature_forward(query_image)

        # 初始化损失（只有 train 模式会真正使用）
        loss_i = 0
        loss_p = 0

        # ---------- 正常参考分支 ----------
        if args.n_shot>0:
            # support_normal 是 (图像, mask) 二元组
            support_n_image, support_n_mask_ = support_normal
            # 正常参考特征：(b,n_shot,N,c)
            support_n_feat = self.feature_forward(support_n_image)
            # 用“正常区域”做匹配（1-mask），得到正常引导分数
            n_pseudo_mask, support_n_mask = self.nn_search(query_feat, support_n_feat, 1-support_n_mask_)
            # 去掉尾维，得到 s_n:(b,1,M)
            s_n = n_pseudo_mask.squeeze(-1)
        
        # ---------- 异常参考分支 ----------
        if args.a_shot>0:
            # support_abnormal 是 (图像, mask) 二元组
            support_a_image, support_a_mask_ = support_abnormal
            # 异常参考特征：(b,a_shot,N,c)
            support_a_feat = self.feature_forward(support_a_image)
            # 把异常 mask 对齐到 token 尺度（第一个返回值此处不用）
            _, support_a_mask = self.nn_search(query_feat, support_a_feat, support_a_mask_)

            if args.n_shot==0: # only abnormal as reference, abtain normal reference from abnormal image
                # 若没有正常参考，则从异常参考的“非异常区域”中抽取伪正常特征
                support_n_feat = support_a_feat.masked_select((1-support_a_mask).bool()).view(support_a_feat.shape[0], 1, -1, support_a_feat.shape[-1]) 
                # 同时基于非异常区域再估计一张正常引导分数图 s_n
                n_pseudo_mask, support_n_mask = self.nn_search(query_feat, support_a_feat, 1-support_a_mask_)
                s_n = n_pseudo_mask.squeeze(-1)

            # RM Module forward
            # 在 support 侧计算“异常-正常”的残差特征
            support_res_feat = self.get_res_feat(support_a_feat, support_n_feat)
            # learnable proxies 从残差中聚合，得到 residual_proxies
            residual_proxies = self.attention_forward(self.rm_ca, self.rm_sa, self.learnable_proxies, support_a_feat, support_res_feat, support_a_mask)

            # AFL Module forward
            # 在 query 侧计算相对正常模板的残差
            query_res_feat = self.get_res_feat(query_feat, support_n_feat)
            # 用 residual_proxies 作为查询，映射得到 anomaly_proxies
            anomaly_proxies_cur = self.attention_forward(self.afl_ca, self.afl_sa, residual_proxies, query_res_feat, query_feat, None)

            # Cross-episode memory retrieval + fusion
            if self.enable_proxy_memory:
                current_conf = self.proxy_confidence(anomaly_proxies_cur, query_res_feat)
                memory_proxies, memory_conf, has_memory = self.retrieve_anomaly_memory(anomaly_proxies_cur)
                if has_memory:
                    anomaly_proxies, _ = self.fuse_anomaly_proxies(anomaly_proxies_cur, memory_proxies, current_conf, memory_conf)
                else:
                    # Bypass fusion when memory is empty to avoid shrinking current proxies.
                    anomaly_proxies = anomaly_proxies_cur
                if self.training:
                    current_epoch = int(getattr(args, "current_epoch", 0))
                    self.update_anomaly_memory(anomaly_proxies_cur, current_conf, current_epoch)
            else:
                anomaly_proxies = anomaly_proxies_cur
            anomaly_proxies = anomaly_proxies.unsqueeze(1)
            
            # query 与 anomaly_proxies 相似度（均值策略）=> 异常引导分数 s_a
            a_out, _ = self.nn_search(query_feat, anomaly_proxies, mode='mean')
            s_a = a_out.squeeze(-1)
        
        # ---------- 处理缺失分支 ----------
        if args.n_shot>0 and args.a_shot==0:
            # 只有正常参考时，用互补关系近似异常分数
            s_a = 1-s_n
        elif args.n_shot==0 and args.a_shot>0:
            # 只有异常参考时，用互补关系近似正常分数
            s_n = 1-s_a
        else:
            # 至少要有一种参考，否则任务无条件约束
            assert args.n_shot>0 or args.a_shot>0, 'n_shot and a_shot should not be both 0'

        # 拼接成 2 通道像素 logits：通道0=正常，通道1=异常
        pixel_level_logits = torch.cat([s_n, s_a], dim=1) # (b, 2, h*w)
        
        # 融合得分：异常分支 + (1-正常分支) 的平均
        a_score = (s_a+(1-s_n))/2

        if mode == 'train':
            # 图像级分数：取 top-k 异常 token 的平均，强调最可疑区域
            a_score_topk = torch.topk(a_score, 20, dim=-1)[0].mean(dim=-1)
            # 构建二分类 logits：[normal_score, abnormal_score]
            image_level_logits = torch.cat([1-a_score_topk, a_score_topk], dim=-1)

            # Image Level
            # 图像级交叉熵损失
            loss_i += self.cross_entropy_loss(image_level_logits, query_label.long())

            # Pixel Level
            # token 序列还原到 2D 网格
            l = int(pixel_level_logits.shape[-1]**0.5)
            pixel_level_logits = rearrange(pixel_level_logits, 'b n (h w) -> b n h w', h=l)
            # 上采样到 query_mask 原始分辨率
            pixel_level_logits = F.interpolate(pixel_level_logits, size=query_mask.shape[-2:], mode='bilinear')
            # 把单通道 mask 变为双通道监督：[正常, 异常]
            query_mask_n = torch.stack([1-query_mask, query_mask], dim=1)
            # 像素级损失：Focal + Dice
            loss_p += self.focal_loss(pixel_level_logits, query_mask_n)
            loss_p += self.dice_loss(pixel_level_logits, query_mask_n)

            # 训练阶段返回 logits 与两项损失
            return image_level_logits, pixel_level_logits, loss_i, loss_p
        
        elif mode == 'test':
            # 测试阶段只返回融合异常分数图
            return a_score
