#!/usr/bin/env python
# -*- coding: utf-8 -*-


import yaml, os
import argparse
from trajdata import UnifiedDataset
from torch.utils.data import DataLoader
from trajdata import AgentType
from tbsim.utils.batch_utils import batch_utils, set_global_batch_type
from collections import defaultdict
def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset(config):
    """根据配置创建数据集"""
    dataset_config = config['dataset']
    
    # 定义数据路径
    data_dirs = {
        dataset_config['name']: dataset_config['data_root']
    }

    # 创建统一数据集
    dataset = UnifiedDataset(
        desired_data=[dataset_config['name'] + "-" + dataset_config['split']],
        data_dirs=data_dirs,
        desired_dt=dataset_config['desired_dt'],
        history_sec=(dataset_config['history_sec'], dataset_config['history_sec']),
        future_sec=(dataset_config['future_sec'], dataset_config['future_sec']),
        agent_interaction_distances= defaultdict(lambda: 30.0),
        only_types=[AgentType.VEHICLE],    # 只考虑车辆
        centric="agent",                   # 修改：从"scene"改为"agent"
        cache_location=dataset_config.get('cache_location', '~/cld_cache'),
        rebuild_cache=dataset_config.get('rebuild_cache', False),
        incl_raster_map=True,
        raster_map_params=dataset_config.get("raster_map_params", {
            "px_per_m": 2.0,
            "map_size_px": 224, 
            "offset_frac_xy": [0.0, 0.0],
            "return_rgb": True,
            "no_map_fill_value": -1.0
        }),
        standardize_data=True,             # 修改：建议改为True，便于agent-centric模式下的学习
        max_neighbor_num=20,               # 新增：限制邻居数量
        ego_only=False,                    # 新增：是否只预测ego vehicle
        verbose=True,
    )
    
    return dataset


def create_dataloader(dataset, config):
    """创建数据加载器"""
    dataloader_config = config['dataloader']
    
    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_config['batch_size'],
        shuffle=True,
        num_workers=dataloader_config['num_workers'],
        collate_fn=dataset.get_collate_fn(return_dict=True)
    )
    
    return dataloader



def main():
    parser = argparse.ArgumentParser(description='从配置文件加载nuScenes数据')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    set_global_batch_type("trajdata")
    # 加载配置
    config = load_config(args.config)
    
    # 创建数据集
    dataset = create_dataset(config)
    print(f"数据集大小: {len(dataset)}个样本")
    
    # 创建数据加载器
    dataloader = create_dataloader(dataset, config)
    
    # 获取一个批次并显示信息
    print("加载第一个批次...")
    for batch in dataloader:
        batch = batch_utils().parse_batch(batch) 
        batch_size = len(batch)
        print(f"批次大小: {batch_size}")


if __name__ == '__main__':
    main() 