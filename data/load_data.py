#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml, os
import argparse
import copy
import pytorch_lightning as pl
from trajdata import UnifiedDataset
from torch.utils.data import DataLoader
from trajdata import AgentType
from tbsim.utils.batch_utils import batch_utils, set_global_batch_type
from collections import defaultdict
import numpy as np
from utils import current_lanes, lane_points, get_agent_lane, get_vector_map
def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_lane_info(x):
    """返回一个包含地图查询功能的字典"""
    vector_map = x.vec_map
    return {
        "map": vector_map,
        # 根据位置查询最近的车道
        "get_lanes_at_point": lambda pos, dist=2.0: vector_map.get_lanes_within(pos, dist),
        
        # 根据位置和朝向查询当前车道
        "get_current_lanes": lambda pos, heading, max_dist=2.0, max_heading_error=np.pi/4: 
            vector_map.get_current_lane(
                np.array([pos[0], pos[1], 0, heading]),
                max_dist=max_dist,
                max_heading_error=max_heading_error
            )
    }

def create_dataset(config, split=None):
    dataset_config = copy.deepcopy(config['dataset'])
    
    # 如果提供了split参数，则覆盖配置中的split
    if split is not None:
        dataset_config['split'] = split
    
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
        agent_interaction_distances=defaultdict(lambda: 50.0),
        only_types=[AgentType.VEHICLE],    # 只考虑车辆
        centric="agent",           
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
        ego_only=True,                    # 新增：是否只预测ego vehicle
        verbose=True,
        incl_vector_map=True,
        vector_map_params={
        "incl_road_lanes": True,
        "incl_road_areas": False,  # 这个参数是必需的
        "incl_ped_crosswalks": False,
        "incl_ped_walkways": False,
        "collate": True,           # 关键：确保这个设置为 True
        "keep_in_memory": True,
    },
        extras={"vector_map":lambda x:x.vec_map}
            
                
    )
    
    return dataset

 

def create_dataloader(dataset, config, shuffle=None):
    """创建数据加载器,允许动态覆盖shuffle参数"""
    dataloader_config = config['dataloader']
    
    # 确定是否打乱数据
    shuffle_data = dataloader_config.get('shuffle', True)
    if shuffle is not None:
        shuffle_data = shuffle
    
    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_config['batch_size'],
        shuffle=shuffle_data,
        num_workers=dataloader_config['num_workers'],
        collate_fn=custom_collate_fn,#dataset.get_collate_fn(return_dict=True),
        drop_last=True
    )
    
    return dataloader


class VAEDataModule(pl.LightningDataModule):
    """处理VAE训练的数据模块"""
    
    def __init__(self, dataset_config):
        super().__init__()
        self.dataset_config = dataset_config
        self.train_dataset = None
        self.val_dataset = None
    
    
    def setup(self, stage=None):
        """准备训练和验证数据集"""
        config = copy.deepcopy(self.dataset_config)
        train_split = config['dataset']['train_split']
        val_split = config['dataset']['val_split']
        self.train_dataset = create_dataset(config, split=train_split)
        self.val_dataset = create_dataset(config, split=val_split)
      
    
    def train_dataloader(self):
        """返回训练数据加载器"""
        return create_dataloader(self.train_dataset, self.dataset_config, shuffle=True)
    
    def val_dataloader(self):
        """返回验证数据加载器"""
        return create_dataloader(self.val_dataset, self.dataset_config, shuffle=False)


def main():
    parser = argparse.ArgumentParser(description='从配置文件加载nuScenes数据')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--test_datamodule', action='store_true', help='测试VAEDataModule')
    args = parser.parse_args()
    set_global_batch_type("trajdata")
    
    # 加载配置
    config = load_config(args.config)
    
    if args.test_datamodule:
        # 测试VAEDataModule
        print("初始化并测试 VAEDataModule...")
        data_module = VAEDataModule(config)
        data_module.setup()
        
        # 输出数据集信息
        print(f"训练数据集大小: {len(data_module.train_dataset)}个样本")
        print(f"验证数据集大小: {len(data_module.val_dataset)}个样本")
        
        # 测试数据加载器
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print("获取训练批次样本...")
        train_batch = next(iter(train_loader))
        train_batch = batch_utils().parse_batch(train_batch)
        print(f"训练批次大小: {len(train_batch)}")
        
        print("获取验证批次样本...")
        val_batch = next(iter(val_loader))
        val_batch = batch_utils().parse_batch(val_batch)
        print(f"验证批次大小: {len(val_batch)}")
    else:
        # 原始测试流程
        # 创建数据集
        dataset = create_dataset(config, split="mini_train")
        print(f"数据集大小: {len(dataset)}个样本")
        
        # 创建数据加载器
        dataloader = create_dataloader(dataset, config)
        
        # 获取一个批次并显示信息
        print("加载第一个批次...")
        for batch in dataloader:
            batch = batch_utils().parse_batch(batch) 
            print(batch['extras']['current_lane'])
            batch_size = len(batch)
            print(f"批次大小: {batch_size}")
            break


if __name__ == '__main__':
    main() 