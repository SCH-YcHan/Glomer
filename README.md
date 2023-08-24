# Glomerulus segmentation in kidney image (PAS)
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

## Study process

## Study result
![Real-time segmentation](https://github.com/SCH-YcHan/Glomer/assets/113504815/e26e0efa-8ca7-4c79-b46c-52e58abde2e6)




