#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data.load_data import VAEDataModule
from trainers.vae_trainer import VAETrainer

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='训练VAE模型')
    parser.add_argument('--dataset_config', type=str, default='configs/datasets/datasets.yaml', help='数据集配置文件路径')
    parser.add_argument('--model_config', type=str, default='configs/models/vae_config.yaml', help='模型配置文件路径')
    parser.add_argument('--training_config', type=str, default='configs/training/vae_training.yaml', help='训练配置文件路径')
    args = parser.parse_args()
    
    # 加载各个配置文件
    dataset_config = load_config(args.dataset_config)
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    
    data_module = VAEDataModule(dataset_config)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    # 创建模型
    if training_config['training'].get('checkpoint'):
        # 从配置文件中指定的检查点加载模型
        model = VAETrainer.load_from_checkpoint(training_config['training']['checkpoint'])
    else:
        # 创建新模型
        model = VAETrainer(
            vae_config=model_config['model'],
            training_config=training_config['training']
        )
    
    # 设置回调函数
    callbacks = []

    # 模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(training_config['training']['output_dir'], 'checkpoints'),
        filename='vae-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        save_top_k=3,
        mode='min'
    )
    callbacks.append(checkpoint_callback)
    
    # 早停回调
    if training_config['training'].get('early_stopping', False):
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=training_config['training'].get('patience', 10),
            mode='min'
        )
        callbacks.append(early_stopping_callback)
    
    # 设置日志记录器
    logger = WandbLogger(
        save_dir=training_config['training']['output_dir'],
        project=training_config['training'].get('wandb_project', 'vae-project'),
        name=training_config['training'].get('wandb_run_name', 'vae-run'),
        log_model=training_config['training'].get('wandb_log_model', True) 
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=training_config['training']['max_epochs'],
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=training_config['training'].get('gradient_clip_val', 0.0),
        log_every_n_steps=training_config['training'].get('log_every_n_steps', 50),
        accelerator=(
        'gpu' if torch.cuda.is_available() else
        'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
        'cpu'
        ),
        devices=1,
        val_check_interval=training_config['training'].get('val_check_interval', 1.0),
        precision="16-mixed" if torch.cuda.is_available() else "32-true"
    )
    
    # 开始训练
    trainer.fit(model, train_loader, val_loader)
    
    # 保存最终模型
    trainer.save_checkpoint(os.path.join(training_config['training']['output_dir'], 'final_model.ckpt'))
    
    print(f"训练完成！最佳模型保存在: {checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    main()
