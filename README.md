# LKHAT
## Hybrid Attention Transformer with Re-parameterized Large Kernel Convolution for Image Super-Resolution

## The codes are based on HAT https://github.com/XPixelGroup/HAT/tree/main

## Environment
- [PyTorch >= 1.7](https://pytorch.org/) **(Recommend **NOT** using torch 1.8!!! It would cause abnormal performance.)**
- [BasicSR == 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 
### Installation
Install Pytorch first.
Then,
```
pip install -r requirements.txt
python setup.py develop
```

## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md).
- The training command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 lkhat/train.py -opt options/train/train_LKHAT_SRx2_from_scratch.yml --launcher pytorch
```
## How To Test
```
python lkhat/test.py -opt options/test/LKHAT_SRx4.yml
```
