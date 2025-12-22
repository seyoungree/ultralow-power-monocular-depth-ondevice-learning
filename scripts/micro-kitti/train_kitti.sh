#!/bin/bash

# Export variable to limit OPENMP threads
export OMP_NUM_THREADS=8

# Create training folders
mkdir 'runs/run0/'
mkdir 'runs/run1/'
mkdir 'runs/run2/'
mkdir 'runs/run3/'
mkdir 'runs/run4/'

# Launch 5 training routines for each setup so to see stability, mean and variance

# 100 percent
CUDA_VISIBLE_DEVICES=0 python train_from_scratch.py --minikitti_path '../../micro-kitti/' --saved_mdl_path 'runs/run0/' --loss_aap 0 --loss_aps 1.0 --percentage_kitti_trainset 100 
CUDA_VISIBLE_DEVICES=0 python train_from_scratch.py --minikitti_path '../../micro-kitti/' --saved_mdl_path 'runs/run1/' --loss_aap 0 --loss_aps 1.0 --percentage_kitti_trainset 100 
CUDA_VISIBLE_DEVICES=0 python train_from_scratch.py --minikitti_path '../../micro-kitti/' --saved_mdl_path 'runs/run2/' --loss_aap 0 --loss_aps 1.0 --percentage_kitti_trainset 100 
CUDA_VISIBLE_DEVICES=0 python train_from_scratch.py --minikitti_path '../../micro-kitti/' --saved_mdl_path 'runs/run3/' --loss_aap 0 --loss_aps 1.0 --percentage_kitti_trainset 100 
CUDA_VISIBLE_DEVICES=0 python train_from_scratch.py --minikitti_path '../../micro-kitti/' --saved_mdl_path 'runs/run4/' --loss_aap 0 --loss_aps 1.0 --percentage_kitti_trainset 100 
