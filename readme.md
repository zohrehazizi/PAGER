# PAGER: Progressive Attribute-Guided Extendable Robust image generation

This repo containes codes for the following [paper](https://arxiv.org/abs/2206.00162):
```sh
@article{azizi2022pager,
  title={PAGER: Progressive Attribute-Guided Extendable Robust Image Generation},
  author={Azizi, Zohreh and Kuo, C-C Jay and others},
  journal={APSIPA Transactions on Signal and Information Processing},
  volume={11},
  number={1},
  year={2022},
  publisher={Now Publishers, Inc.}
}
```

## Installation
- Run the folowing lines to create a conda environment: 
```sh
conda init bash
source ~/.bashrc
conda create -n PAGER python==3.8.5
conda activate PAGER
```
- Run `bash install.sh` to install required packages.

## Dataset preparation
The code is tested for [MNIST](http://yann.lecun.com/exdb/mnist/), [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) and [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
You must download the datasets and place them in their corresponding folders in the `datasets` folder.

## Usage
- If you use USC HPC server, use the following commands to setup a GPU node:
```sh
salloc --time=48:00:00 --ntasks=1 --cpus-per-task=1 --mem=180GB --gres=gpu:v100:1 --partition=gpu
```
- Load the following cuda and cudnn versions and setup the environment:
```sh
conda deactivate
module load gcc/8.3.0
module load cuda/10.1.243
module load cudnn/7.6.5.32-10.1
module load anaconda3
conda activate PAGER
```
- Run `main_celeba.py` to train and generate for CelebA dataset.
- Run `main_mnist_fashion.py` to train and generate for mnist or fashion-mnist datasets. 

