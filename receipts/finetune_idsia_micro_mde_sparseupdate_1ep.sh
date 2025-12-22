#!/bin/bash

# Export variable to limit OPENMP threads
export OMP_NUM_THREADS=8
# Save experiment name and 
EXPERIMENT_NAME="finetune_idsia_micro_mde_sparseupdate_1ep"
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
CUDA_VISIBLE_DEVICES=0 python finetune_pretrained_on_idsia-micro-mde.py --idsiadepth_path '../../idsia-micro-mde/' --saved_mdl_path "$EXPERIMENT_NAME/run0/" --downsample_prediction_or_upsample_label 'UPSAMPLE_LABEL' --upsample_strategy 'bilinear' --align_cam_tof_fov 0 --set_tof_max_depth_to_invalid 1 --epochs 1 --su_update_encoder 0 --su_update_dec0 1 --su_update_dec1 0 --su_update_dec2 0 
CUDA_VISIBLE_DEVICES=0 python finetune_pretrained_on_idsia-micro-mde.py --idsiadepth_path '../../idsia-micro-mde/' --saved_mdl_path "$EXPERIMENT_NAME/run1/" --downsample_prediction_or_upsample_label 'UPSAMPLE_LABEL' --upsample_strategy 'bilinear' --align_cam_tof_fov 0 --set_tof_max_depth_to_invalid 1 --epochs 1 --su_update_encoder 0 --su_update_dec0 1 --su_update_dec1 0 --su_update_dec2 0 
CUDA_VISIBLE_DEVICES=0 python finetune_pretrained_on_idsia-micro-mde.py --idsiadepth_path '../../idsia-micro-mde/' --saved_mdl_path "$EXPERIMENT_NAME/run2/" --downsample_prediction_or_upsample_label 'UPSAMPLE_LABEL' --upsample_strategy 'bilinear' --align_cam_tof_fov 0 --set_tof_max_depth_to_invalid 1 --epochs 1 --su_update_encoder 0 --su_update_dec0 1 --su_update_dec1 0 --su_update_dec2 0 
CUDA_VISIBLE_DEVICES=0 python finetune_pretrained_on_idsia-micro-mde.py --idsiadepth_path '../../idsia-micro-mde/' --saved_mdl_path "$EXPERIMENT_NAME/run3/" --downsample_prediction_or_upsample_label 'UPSAMPLE_LABEL' --upsample_strategy 'bilinear' --align_cam_tof_fov 0 --set_tof_max_depth_to_invalid 1 --epochs 1 --su_update_encoder 0 --su_update_dec0 1 --su_update_dec1 0 --su_update_dec2 0 
CUDA_VISIBLE_DEVICES=0 python finetune_pretrained_on_idsia-micro-mde.py --idsiadepth_path '../../idsia-micro-mde/' --saved_mdl_path "$EXPERIMENT_NAME/run4/" --downsample_prediction_or_upsample_label 'UPSAMPLE_LABEL' --upsample_strategy 'bilinear' --align_cam_tof_fov 0 --set_tof_max_depth_to_invalid 1 --epochs 1 --su_update_encoder 0 --su_update_dec0 1 --su_update_dec1 0 --su_update_dec2 0 

# Run the 5 testing routines
CUDA_VISIBLE_DEVICES=0 python test_idsia-micro-mde.py --idsiadepth_path '../../idsia-micro-mde/' --saved_mdl_path "$EXPERIMENT_NAME/run0/" --log_dir "$EXPERIMENT_NAME/run0/" --test_visual_output 'No' --set_tof_max_depth_to_invalid 1
CUDA_VISIBLE_DEVICES=0 python test_idsia-micro-mde.py --idsiadepth_path '../../idsia-micro-mde/' --saved_mdl_path "$EXPERIMENT_NAME/run1/" --log_dir "$EXPERIMENT_NAME/run1/" --test_visual_output 'No' --set_tof_max_depth_to_invalid 1
CUDA_VISIBLE_DEVICES=0 python test_idsia-micro-mde.py --idsiadepth_path '../../idsia-micro-mde/' --saved_mdl_path "$EXPERIMENT_NAME/run2/" --log_dir "$EXPERIMENT_NAME/run2/" --test_visual_output 'No' --set_tof_max_depth_to_invalid 1
CUDA_VISIBLE_DEVICES=0 python test_idsia-micro-mde.py --idsiadepth_path '../../idsia-micro-mde/' --saved_mdl_path "$EXPERIMENT_NAME/run3/" --log_dir "$EXPERIMENT_NAME/run3/" --test_visual_output 'No' --set_tof_max_depth_to_invalid 1
CUDA_VISIBLE_DEVICES=0 python test_idsia-micro-mde.py --idsiadepth_path '../../idsia-micro-mde/' --saved_mdl_path "$EXPERIMENT_NAME/run4/" --log_dir "$EXPERIMENT_NAME/run4/" --test_visual_output 'No' --set_tof_max_depth_to_invalid 1

# Move the obtained folder to the main one
echo "Training completed, moving results"
mv "$EXPERIMENT_NAME" "$TARGET_PATH"
cd $TARGET_PATH
