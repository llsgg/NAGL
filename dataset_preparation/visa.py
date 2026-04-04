import os
import json
import pandas as pd


class VisASolver(object):
    # VisA 数据集类别名
    CLSNAMES = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        'pcb4', 'pipe_fryum',
    ]

    def __init__(self, root='data/visa', meta_output=None):
        # 数据根目录
        self.root = root
        # meta.json 输出路径
        self.meta_path = f'{meta_output}/meta.json' if meta_output else f'{root}/meta.json'
        # VisA 这里固定 train/test 两个 split
        self.phases = ['train', 'test']
        # 从官方 split csv 读取样本描述
        # 典型列：[object, split, label, image, mask]
        self.csv_data = pd.read_csv(f'{root}/split_csv/1cls.csv', header=0)

    def run(self):
        # 读取列名，避免硬编码字符串时写错
        columns = self.csv_data.columns  # [object, split, label, image, mask]
        # 初始化输出结构：info['train'] / info['test']
        info = {phase: {} for phase in self.phases}
        # 统计 test 集中正常/异常样本数
        anomaly_samples = 0
        normal_samples = 0
        # 逐类别处理
        for cls_name in self.CLSNAMES:
            # 先筛出当前类别
            cls_data = self.csv_data[self.csv_data[columns[0]] == cls_name]
            # 再按 train/test 分开
            for phase in self.phases:
                cls_info = []
                cls_data_phase = cls_data[cls_data[columns[1]] == phase]
                # 重排索引，便于后续 loc[idx] 连续访问
                cls_data_phase.index = list(range(len(cls_data_phase)))
                # 逐行生成 meta 记录
                for idx in range(cls_data_phase.shape[0]):
                    data = cls_data_phase.loc[idx]
                    # label='anomaly' 视为异常样本
                    is_abnormal = True if data[2] == 'anomaly' else False
                    info_img = dict(
                        # 这里 csv 里通常已经是相对路径，直接使用
                        img_path=data[3],
                        # 异常样本填 mask 路径，正常样本置空
                        mask_path=data[4] if is_abnormal else '',
                        # 类别名
                        cls_name=cls_name,
                        # VisA 在本脚本里不使用 specie，留空字符串
                        specie_name='',
                        # 图像级标签：异常=1，正常=0
                        anomaly=1 if is_abnormal else 0,
                    )
                    cls_info.append(info_img)
                    # 仅统计 test 阶段样本数量
                    if phase == 'test':
                        if is_abnormal:
                            anomaly_samples = anomaly_samples + 1
                        else:
                            normal_samples = normal_samples + 1
                # 保存当前 phase+class 的样本列表
                info[phase][cls_name] = cls_info
        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        # 写出 meta.json
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        # 打印 test 集统计，方便核查
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)


if __name__ == '__main__':
    # 用法示例：按你的本地路径修改后运行
    runner = VisASolver(
        root='/data2/zhangheyao/PromptAD-master/data/visa',
        meta_output='../dataset/meta_json/visa'
    )
    # 执行元数据构建
    runner.run()
