# 项目环境配置指南

该文档详细记录了如何设置conda环境以及安装所需的依赖包。

## 环境要求

* Python 3.10
* CUDA 1.13.1

## 创建conda环境

### 1. 创建新环境

```bash
# 创建一个名为cld+的Python 3.10环境
conda create --name cld+ python=3.10 -y
```

### 2. 激活环境

```bash
# 激活环境
conda activate cld+
```

## 安装依赖包

### 安装l5kit (无依赖)

l5kit要求PyTorch版本在1.5.0到2.0.0之间，和lightning冲突,所以先不安装依赖

```bash
pip install l5kit --no-deps 
```

### 安装 Pytorch_Lightning: 2.2版本

```bash
conda install pytorch-lightning=2.2.0 -c conda-forge 
```

### 安装 PyTorch

```bash
linux:
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

osx:
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
```
### 检查numpy版本
```bash
conda install numpy=1.23.5
```

### 安装 Trajdata
```bash
pip install "trajdata[nusc]"
```
### 安装 tbsim
```bash
cd third_party/tbsim
pip install -e .
```
### Matplotlib 检查版本(应该是3.5.3, 过高版本和nuscenes不兼容)
```bash
pip uninstall -y matplotlib
pip install matplotlib==3.5.3
```

## 验证安装

创建一个简单的测试脚本`test_env.py`：

```python
import torch
import l5kit
import pytorch_lightning as pl

print(f"PyTorch版本: {torch.__version__}")
print(f"l5kit版本: {l5kit.__version__}")
print(f"PyTorch Lightning版本: {pl.__version__}")
```

运行脚本验证所有包都已正确安装：

```bash
python test_env.py
```

## 环境问题解决

### PATH环境变量

如果遇到`pip install l5kit`出现报错,可能是pip安装错误

```bash
# 检查pip路径
which pip

#正确输出
/home/visier/anaconda3/envs/cld+/bin/pip
