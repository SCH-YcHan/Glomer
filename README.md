# Glomerulus segmentation in large kidney image
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
## Study process
### 1. Make Pre-trained model (using [HuBMAP dataset](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature))
- Data preprocessing & Cross validation & Create annotation ([refer](https://www.kaggle.com/code/ammarnassanalhajali/hubmap-2023-k-fold-cv-coco-dataset-generator))
- Make config file (mmdet format, [refer](https://www.kaggle.com/code/andtaichi/hubmap-mmdet-ver3-0-0-training)) & fine tuning the detection and segmentation model
- Compare single class (only glomerulus) model with multi class (plus alpha) model
- Evaluate model predictions and performance
### 2. Glomerulus detection with Pre-trained model in large kidney image (59342 x 114316, 2.3 GB) 
- Using [SAHI](https://github.com/obss/sahi) (Slicing Aided Hyper Inference)
- But... very bad performance :(
- Need to explore other methods
## Study result
### Glomerulus segmentation (HuBMAP Dataset)
![image](https://github.com/SCH-YcHan/Glomer/assets/113504815/14bd08fd-62c7-4097-a3d6-130d00584bf2)
![image](https://github.com/SCH-YcHan/Glomer/assets/113504815/138dc0df-81f9-4515-8b53-00c4fd4a8c8f)



