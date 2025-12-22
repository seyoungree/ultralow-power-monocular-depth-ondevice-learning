#!/bin/bash

# Export variable to limit OPENMP threads
export OMP_NUM_THREADS=8
# Save experiment name and 
EXPERIMENT_NAME="train_on_kitti"
export EXPERIMENT_NAME
# Save target folder to store experiment runs
cd ..
export TARGET_PATH=$(pwd)

# Go to the experiments folder
cd scripts/micro-kitti/ || exit 1

# Create the folder
mkdir -p "$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_NAME/run0/"
mkdir -p "$EXPERIMENT_NAME/run1/"
mkdir -p "$EXPERIMENT_NAME/run2/"
mkdir -p "$EXPERIMENT_NAME/run3/"
mkdir -p "$EXPERIMENT_NAME/run4/"


# Run the 5 training routines
CUDA_VISIBLE_DEVICES=0 python train_from_scratch.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run0/" --percentage_kitti_trainset 100 
CUDA_VISIBLE_DEVICES=0 python train_from_scratch.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run1/" --percentage_kitti_trainset 100 
CUDA_VISIBLE_DEVICES=0 python train_from_scratch.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run2/" --percentage_kitti_trainset 100 
CUDA_VISIBLE_DEVICES=0 python train_from_scratch.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run3/" --percentage_kitti_trainset 100 
CUDA_VISIBLE_DEVICES=0 python train_from_scratch.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run4/" --percentage_kitti_trainset 100 

# Run the 5 testing routines
CUDA_VISIBLE_DEVICES=0 python test.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run0/" --log_dir "$EXPERIMENT_NAME/run0/" --test_visual_output 'No' 
CUDA_VISIBLE_DEVICES=0 python test.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run1/" --log_dir "$EXPERIMENT_NAME/run1/" --test_visual_output 'No' 
CUDA_VISIBLE_DEVICES=0 python test.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run2/" --log_dir "$EXPERIMENT_NAME/run2/" --test_visual_output 'No' 
CUDA_VISIBLE_DEVICES=0 python test.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run3/" --log_dir "$EXPERIMENT_NAME/run3/" --test_visual_output 'No' 
CUDA_VISIBLE_DEVICES=0 python test.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run4/" --log_dir "$EXPERIMENT_NAME/run4/" --test_visual_output 'No' 


# Move the obtained folder to the main one
echo "Training completed, moving results"
mv "$EXPERIMENT_NAME" "$TARGET_PATH"
cd $TARGET_PATH
