r""" Few-Shot Anomaly Detection Dataset """

import os
import json
import random   

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import PIL.Image as Image
import numpy as np

class FSDataset(Dataset):
    # Few-shot 异常检测数据集：
    # 每次 __getitem__ 返回一个 episode（query + normal support + abnormal support）
    def __init__(self,
                 data_root:str ='/data/datasets', 
                 meta_root:str = None,
                 data_mode:str = 'mvtec_visa',
                 data_name_json:str ='meta.json',
                 fold:int =0,
                 split:str ='train',
                 shot:list =[5, 1],
                 transform:object =None, 
                 choice=500):
        # eval/test 都使用 val 的类别划分
        self.split = 'val' if split in ['eval', 'test'] else 'train'
        # 图像根目录
        self.data_root = data_root
        # meta.json 根目录（默认与 data_root 相同）
        self.meta_root = meta_root if meta_root else data_root
        # 支持两种模式：mvtec+visa，或 Real-IAD
        data = ['mvtec', 'visa'] if data_mode == 'mvtec_visa' else ['Real-IAD/realiad_1024_unzip']
        # data = ['mvtec', 'visa'] if data_mode == 'mvtec_visa' else ['realiad']
        self.data_mode = data_mode
        # 元数据文件名，通常是 meta.json
        self.data_name_json = data_name_json

        # 训练时一个 epoch 取多少个 episode（__len__ 使用）
        self.choice = choice
        # self.nfolds = 2
        # 交叉验证 fold id
        self.fold = fold
        # n_shot: 正常 support 数；a_shot: 异常 support 数
        self.n_shot, self.a_shot = shot

        # 图像预处理（ToTensor/Resize/Normalize 等）
        self.transform = transform

        # 加载并整理 metadata 到内存
        self.initialize(data)

        # 根据 split/fold 得到当前可采样类别 id
        self.class_ids = self.build_class_ids()
 
    def initialize(self, data):
        # Generate metadata
        # metadata 结构：
        # metadata[product]['normal'/'abnormal'][specie_name] = [sample_info, ...]
        self.metadata = {}
        # 所有产品类别名称列表（排序后）
        self.all_product = []
        for d in data:
            data_path = os.path.join(self.data_root, d)
            meta_path = os.path.join(self.meta_root, d)
            with open(os.path.join(meta_path, self.data_name_json)) as f:
                # data_info.update(json.load(f)['test'])
                data_info = json.load(f)

            data_info = data_info['test']
            products = list(data_info.keys())
            products.sort()
            self.all_product += products

            for product in products:
                # 为每个 product 预建 normal/abnormal 容器
                self.metadata.setdefault(product, {'normal': {}, 'abnormal': {}})
                
                for sample in data_info[product]:
                    # 每个样本保存图像路径、mask 路径、是否异常
                    sample_info = {
                        'img_path': os.path.join(data_path, sample['img_path']),
                        'mask_path': os.path.join(data_path, sample['mask_path']),
                        'anomaly': sample['anomaly']
                    }
                    
                    if sample['anomaly']:
                        product_type = 'abnormal'
                        # 若 specie_name 为空，异常默认记为 bad
                        if sample['specie_name'] == '':
                            sample['specie_name'] = 'bad'
                    else:
                        product_type = 'normal'
                        # 若 specie_name 为空，正常默认记为 good
                        if sample['specie_name'] == '':
                            sample['specie_name'] = 'good'
                    # 按 product_type + specie_name 分桶存储
                    self.metadata[product][product_type].setdefault(sample['specie_name'], []).append(sample_info)
        
        # 总类别数
        self.nclass = len(self.all_product)

    def __len__(self):
        # return self.choice
        # 训练：返回 choice；验证：固定 2000 个 episode 做评估
        return self.choice if self.split == 'train' else 2000
    
    def mask_list_transform(self, mask_list, img_size):
        # 把 mask 列表统一 resize 到目标尺寸，再堆叠成张量
        mask_tensor = []
        for mask in mask_list:
            mask_tensor.append(F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), img_size, mode='nearest').squeeze())
        return torch.stack(mask_tensor)
    
    def image_list_transform(self, image_list):
        # 对图像列表逐个应用 transform，再堆叠
        image_tensor = []
        for image in image_list:
            image_tensor.append(self.transform(image))
        return torch.stack(image_tensor)

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # tuple: (image_list, mask_list, anomaly_list)
        # 这里并不按 idx 读取固定样本，而是每次随机采样一个 episode
        query_tuple, support_normal_tuple, support_abnormal_tuple, sample_product = self.load_frame()

        # query: 一般只有 1 张图
        query_image = self.image_list_transform(query_tuple[0]) # [1, C, H, W]
        query_mask = self.mask_list_transform(query_tuple[1], query_image.shape[-2:]) # [1, H, W]
        query_data = [query_image, query_mask]

        # 先给 support 占位（当 n_shot/a_shot 为 0 时会保持占位值）
        support_normal_data = [0, 0]
        support_abnormal_data = [0, 0]
        
        # 正常 support
        if self.n_shot:
            support_normal_imgs = self.image_list_transform(support_normal_tuple[0])
            support_normal_masks = self.mask_list_transform(support_normal_tuple[1], support_normal_imgs.size()[-2:])
            support_normal_data = [support_normal_imgs, support_normal_masks]
        
        # 异常 support
        if self.a_shot:
            support_abnormal_imgs = self.image_list_transform(support_abnormal_tuple[0])
            support_abnormal_masks = self.mask_list_transform(support_abnormal_tuple[1], support_abnormal_imgs.size()[-2:])
            support_abnormal_data = [support_abnormal_imgs, support_abnormal_masks]

        # 按训练脚本约定的字段返回
        ret_dict = {
            'query': query_data,
            # query_tuple[2] 是 anomaly_list（长度为1），训练中会取 [0] 得到图像级标签
            'image_level_label': query_tuple[2],
            'support_normal': support_normal_data,
            'support_abnormal': support_abnormal_data,
            'sample_product': sample_product
        }
        
        return ret_dict

    def build_class_ids(self):
        if self.data_mode == 'mvtec_visa': # mvtec and visa
            # fold=0: 0~14 做验证，其他做训练
            # fold=1: 15~26 做验证，其他做训练
            class_ids_val = range(0, 15) if self.fold == 0 else range(15, 27)
            class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        else: # RealIAD
            self.nfolds = 2
            # number of product categories
            nclass_trn = self.nclass // self.nfolds
            class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
            class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        class_trn = [self.all_product[i] for i in class_ids_trn]
        class_val = [self.all_product[i] for i in class_ids_val]
        # 根据 split 选择当前可见类别 id
        class_ids = class_ids_trn if self.split == 'train' else class_ids_val

        msg = f'Train classes: {class_trn}' if self.split == 'train' else  f'Val classes: {class_val} \n'
        print(msg)

        # 保存当前 split 的产品名列表，评估按类别统计时会用到
        self.products = class_trn if self.split == 'train' else class_val

        return class_ids
    
    def random_sample(self, sample_list, num):
        # 在给定样本池中无放回随机采样 num 个
        selected_idx = random.sample(range(len(sample_list)), num)
        selected_sample = [sample_list[i] for i in selected_idx]
        return selected_idx, selected_sample
    
    def get_sample_info(self, sample_dict):
        # 从样本字典中提取核心字段
        img_path = sample_dict['img_path']
        mask_path = sample_dict['mask_path']
        anomaly = sample_dict['anomaly']
        return img_path, mask_path, anomaly
    
    def read_mask(self, anomaly, mask_path, imsize):
        # 正常样本直接返回全 0 mask
        if not anomaly:
            mask = torch.zeros(imsize)
        else:
            # 异常样本从磁盘读取灰度 mask 并二值化
            mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
            mask[mask>0.5] = 1
        return mask
    
    def read_data(self, data_list):
        # 把样本列表读取成三个并行列表：图像、mask、异常标记
        image_list = []
        mask_list = []
        anomaly_list = []
        for data in data_list:
            data_path, mask_path, anomaly = self.get_sample_info(data)
            image_list.append(Image.open(data_path).convert('RGB'))
            mask_list.append(self.read_mask(anomaly, mask_path, image_list[-1].size))
            anomaly_list.append(anomaly)
        return image_list, mask_list, anomaly_list

    def load_frame(self):

        # 1) 先随机抽一个类别（product）
        sample_product_idx = np.random.choice(self.class_ids, 1)[0]
        sample_product = self.all_product[sample_product_idx]

        # 2) 采样 normal support：先随机选一个正常缺陷子类（specie），再采样 n_shot 张
        support_normal_specie_type = np.random.choice(list(self.metadata[sample_product]['normal'].keys()), 1)[0]
        support_normal_idx, support_normal = self.random_sample(self.metadata[sample_product]['normal'][support_normal_specie_type], self.n_shot)
        # 3) 采样 abnormal support：同理采样 a_shot 张
        support_abnormal_specie_type = np.random.choice(list(self.metadata[sample_product]['abnormal'].keys()), 1)[0]
        if len(self.metadata[sample_product]['abnormal'][support_abnormal_specie_type])<self.a_shot:
            print(self.metadata[sample_product]['abnormal'][support_abnormal_specie_type])
        support_abnormal_idx, support_abnormal = self.random_sample(self.metadata[sample_product]['abnormal'][support_abnormal_specie_type], self.a_shot)

        # 4) 采样 query：随机决定 normal 或 abnormal
        query_type = np.random.choice(['normal', 'abnormal'], 1)[0]
        # query 的 specie 与对应 support 的 specie 对齐（保证语义一致）
        query_specie_type = support_normal_specie_type if query_type == 'normal' else support_abnormal_specie_type
        query_idx, query_data = self.random_sample(self.metadata[sample_product][query_type][query_specie_type], 1)
        
        # 5) 防止 query 恰好与 support 重复
        if ((query_idx in support_normal_idx and query_type == 'normal') or 
            (query_idx in support_abnormal_idx and query_type == 'abnormal')):
            self.load_frame()

        # 6) 真正读取 query 图像/mask/标签
        query_tuple = self.read_data(query_data)
        
        # 7) 读取 normal support
        support_normal_tuple = (0, 0, 0)
        if self.n_shot > 0:
            support_normal_tuple = self.read_data(support_normal)

        # 8) 读取 abnormal support
        support_abnormal_tuple = (0, 0, 0)
        if self.a_shot > 0:
            support_abnormal_tuple = self.read_data(support_abnormal)

        # 返回一个 episode 所需的完整信息
        return query_tuple, support_normal_tuple, support_abnormal_tuple, sample_product