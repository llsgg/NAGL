import os
import json


class MVTecSolver(object):
    # MVTec AD 的 15 个类别
    CLSNAMES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    ]

    def __init__(self, root='data/mvtec', meta_output=None):
        # 数据根目录（应包含每个类别文件夹）
        self.root = root
        # 输出 meta.json 的路径；若未指定输出目录，则默认写到 root/meta.json
        self.meta_path = f'{meta_output}/meta.json' if meta_output else f'{root}/meta.json'

    def run(self):
        # 最终写出的结构：info['train'][cls_name] / info['test'][cls_name]
        info = dict(train={}, test={})
        # 统计 test 集中正常/异常样本数量，方便核对数据是否完整
        anomaly_samples = 0
        normal_samples = 0
        # 逐类别遍历
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            # train/test 两个阶段分别构建元数据
            for phase in ['train', 'test']:
                cls_info = []
                # MVTec 目录层级：{cls}/{phase}/{specie}/xxx.png
                species = os.listdir(f'{cls_dir}/{phase}')
                # 逐缺陷类型（specie）处理：good、scratch、hole...
                for specie in species:
                    # good 视为正常，其余 specie 视为异常
                    is_abnormal = True if specie not in ['good'] else False
                    # 图像文件列表
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    # 异常样本才有像素级 mask，路径在 ground_truth 下
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    # 排序后按索引对齐（依赖官方命名规则）
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    # 逐图生成一条 meta 记录
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            # 相对 data_root 的图像路径
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            # 异常样本填写 mask 路径，正常样本留空字符串
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal else '',
                            # 类别名（产品名）
                            cls_name=cls_name,
                            # 缺陷子类名（good 或具体缺陷类型）
                            specie_name=specie,
                            # 图像级标签：异常=1，正常=0
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                        # 仅在 test 阶段做样本统计
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples = anomaly_samples + 1
                            else:
                                normal_samples = normal_samples + 1
                # 保存当前 phase+class 的所有样本记录
                info[phase][cls_name] = cls_info
        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        # 写出格式化 JSON，便于人工检查
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        # 打印 test 集统计信息
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)

if __name__ == '__main__':
    # 用法示例：修改 root/meta_output 为你的本地路径后运行
    runner = MVTecSolver(
        root='/data2/zhangheyao/PromptAD-master/data/mvtec',
        meta_output='../dataset/meta_json/mvtec'
    )
    # 执行元数据构建
    runner.run()
