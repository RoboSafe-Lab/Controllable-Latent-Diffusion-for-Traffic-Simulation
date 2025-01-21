# HazardForge

**Dynamic and Realistic Vehicle Scenario Generation for Autonomous Safety Testing**

Codebase for **HazardForge**, focusing on the generation of controllable and dynamic hazardous driving scenarios for autonomous vehicle testing. This framework leverages reinforcement learning and diffusion models to create complex traffic situations, pushing the limits of safety-critical driving environments.
## Installation
### Basic (mainly based on tbsim)
Create conda environment (Note nuplan-devkit needs `python>=3.9` so the virtual environment with python version 3.9 needs to be created instead of python 3.8.)
```angular2html
conda env create -f environment.yml
conda activate hf
cd ~
git clone https://github.com/RoboSafe-Lab/HazardForge.git
cd HazardForge
pip install -e .

pip install numpy==2.0.2
pip install pytorch_lightning==2.1.0
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

```
Install a customized version of `trajdata`
```angular2html
cd ..
git clone https://github.com/AIasd/trajdata.git
cd trajdata
pip install -r trajdata_requirements.txt
pip install -e .
```
Install `Pplan`
```angular2html
cd ..
git clone https://github.com/NVlabs/spline-planner.git Pplan
cd Pplan
pip install -e .
```

## Quick start
### 1. Obtain dataset(s)
We currently support the nuScenes [dataset](https://www.nuscenes.org/nuscenes).