# Glomerulus segmentation in large kidney images
## Env setting (Local)
```
OS: Windows 10 Pro
CPU: Intel(R) Core(TM) i9-10900X 
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA version: 11.7
CuDNN version: 8.4.0
Workstation: Anaconda3
```
```
# Anaconda env setting
conda create -n HuBMAP python=3.10
activate HuBMAP
conda install pytorch torchvision torchaudio-cuda=11.7 -c pytorch -c nvidia
conda install jupyter notebook
```
## Study process
### 1. Make Pre-trained model (using HuBMAP dataset)
- Make config file (mmdet format) & fine tuning the detection and segmentation model
- Compare single class(only glomerulus) model with multi class(plus alpha) model
- Evaluate model predictions and performance
### 2. d
- dd
## Result
### Glomerulus segmentation result (Kaggle HuBMAP Competition Dataset)
![image](https://github.com/SCH-YcHan/Glomer/assets/113504815/14bd08fd-62c7-4097-a3d6-130d00584bf2)
![image](https://github.com/SCH-YcHan/Glomer/assets/113504815/138dc0df-81f9-4515-8b53-00c4fd4a8c8f)



