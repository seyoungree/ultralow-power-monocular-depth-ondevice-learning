# Training and On-Device Learning simulation scripts

This folder contains the training scripts and utilities to reproduce the result on the public datasets and on IDSIA-μMDE. The main folder is: 

- [pre-train-and-odl](./pre-train-and-odl/), which contains all the training and testing scripts to pre-train μPyD-Net on TartanAir and fine-tune it on KITTI, NYUv2 and IDSIA-μMDE, comprising also the simulation of the 8x8 labels for the open-access datasets. 

We also provide two additional folders, containing the training scripts on the open-access datasets used to evaluate μPyD-Net for tiny monocular depth estimation: 

- the [micro-kitti](./micro-kitti/) folder contains the script to train from scratch and test μPyD-Net on the 48x48 KITTI dataset
- the [micro-nyuv2](./micro-nyuv2/) folder contains the script to train from scratch and test μPyD-Net on the 48x48 NYUv2 dataset

## Pre-made training receipts

To reproduce the results of the paper, pre-made training scripts are provided in every folder. 
