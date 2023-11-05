# Real-time glomerulus segmentation in kidney image (PAS)
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
```
# git clone this repository
git clone https://github.com/SCH-YcHan/Glomer.git 
```
```
# git clone mmsegmentation
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```
## Dataset

## Usage model [[Xu et al. 2023](https://github.com/XuJiacong/PIDNet)]
![image](https://github.com/SCH-YcHan/Glomer/assets/113504815/fbe6d90a-cfec-4e2c-ae7b-cca9f61f7387)

## Study result
### 1. Segmentation (Green: Ground truth, Red: Pred seg map)
![image](https://github.com/SCH-YcHan/Glomer/assets/113504815/fdc4bc1e-8e92-4341-a58b-42c8d75dc0ad)

### 2. Real-time segmentation
![Real-time segmentation](https://github.com/SCH-YcHan/Glomer/assets/113504815/e26e0efa-8ca7-4c79-b46c-52e58abde2e6)




