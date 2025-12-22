#!/bin/bash

# Export variable to limit OPENMP threads
export OMP_NUM_THREADS=8

# Launch 5 training routines for each setup so to see stability, mean and variance

# 100 percent
CUDA_VISIBLE_DEVICES=0 python test.py --minikitti_path '../../micro-kitti/' --saved_mdl_path 'runs/run0/' --log_dir 'runs/run0/' --test_visual_output 'No' 
CUDA_VISIBLE_DEVICES=0 python test.py --minikitti_path '../../micro-kitti/' --saved_mdl_path 'runs/run1/' --log_dir 'runs/run1/' --test_visual_output 'No' 
CUDA_VISIBLE_DEVICES=0 python test.py --minikitti_path '../../micro-kitti/' --saved_mdl_path 'runs/run2/' --log_dir 'runs/run2/' --test_visual_output 'No' 
CUDA_VISIBLE_DEVICES=0 python test.py --minikitti_path '../../micro-kitti/' --saved_mdl_path 'runs/run3/' --log_dir 'runs/run3/' --test_visual_output 'No' 
CUDA_VISIBLE_DEVICES=0 python test.py --minikitti_path '../../micro-kitti/' --saved_mdl_path 'runs/run4/' --log_dir 'runs/run4/' --test_visual_output 'No' 
