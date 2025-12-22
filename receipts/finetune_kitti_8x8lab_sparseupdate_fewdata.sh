#!/bin/bash

# Export variable to limit OPENMP threads
export OMP_NUM_THREADS=8
# Save experiment name and 
EXPERIMENT_NAME="finetune_kitti_8x8lab_sparseupdate_fewdata"
export EXPERIMENT_NAME
# Save target folder to store experiment runs
cd ..
export TARGET_PATH=$(pwd)

# Go to the experiments folder
cd scripts/pre-train-and-odl || exit 1

# Create the folder
mkdir -p "$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_NAME/run0/"
mkdir -p "$EXPERIMENT_NAME/run1/"
mkdir -p "$EXPERIMENT_NAME/run2/"
mkdir -p "$EXPERIMENT_NAME/run3/"
mkdir -p "$EXPERIMENT_NAME/run4/"


# Run the 5 training routines
CUDA_VISIBLE_DEVICES=0 python finetune_pretrained_on_kitti.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run0/" --su_update_encoder 0 --su_update_dec0 1 --su_update_dec1 0 --su_update_dec2 0 --percentage_kitti_trainset 20 --proxy_disparity_resolution '8x8' --epochs 120 
CUDA_VISIBLE_DEVICES=0 python finetune_pretrained_on_kitti.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run1/" --su_update_encoder 0 --su_update_dec0 1 --su_update_dec1 0 --su_update_dec2 0 --percentage_kitti_trainset 20 --proxy_disparity_resolution '8x8' --epochs 120 
CUDA_VISIBLE_DEVICES=0 python finetune_pretrained_on_kitti.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run2/" --su_update_encoder 0 --su_update_dec0 1 --su_update_dec1 0 --su_update_dec2 0 --percentage_kitti_trainset 20 --proxy_disparity_resolution '8x8' --epochs 120 
CUDA_VISIBLE_DEVICES=0 python finetune_pretrained_on_kitti.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run3/" --su_update_encoder 0 --su_update_dec0 1 --su_update_dec1 0 --su_update_dec2 0 --percentage_kitti_trainset 20 --proxy_disparity_resolution '8x8' --epochs 120 
CUDA_VISIBLE_DEVICES=0 python finetune_pretrained_on_kitti.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run4/" --su_update_encoder 0 --su_update_dec0 1 --su_update_dec1 0 --su_update_dec2 0 --percentage_kitti_trainset 20 --proxy_disparity_resolution '8x8' --epochs 120 

# Run the 5 testing routines
CUDA_VISIBLE_DEVICES=0 python test_kitti.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run0/" --log_dir "$EXPERIMENT_NAME/run0/" --test_visual_output 'No'
CUDA_VISIBLE_DEVICES=0 python test_kitti.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run1/" --log_dir "$EXPERIMENT_NAME/run1/" --test_visual_output 'No'
CUDA_VISIBLE_DEVICES=0 python test_kitti.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run2/" --log_dir "$EXPERIMENT_NAME/run2/" --test_visual_output 'No'
CUDA_VISIBLE_DEVICES=0 python test_kitti.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run3/" --log_dir "$EXPERIMENT_NAME/run3/" --test_visual_output 'No'
CUDA_VISIBLE_DEVICES=0 python test_kitti.py --minikitti_path '../../micro-kitti/' --saved_mdl_path "$EXPERIMENT_NAME/run4/" --log_dir "$EXPERIMENT_NAME/run4/" --test_visual_output 'No'


# Move the obtained folder to the main one
echo "Training completed, moving results"
mv "$EXPERIMENT_NAME" "$TARGET_PATH"
cd $TARGET_PATH
