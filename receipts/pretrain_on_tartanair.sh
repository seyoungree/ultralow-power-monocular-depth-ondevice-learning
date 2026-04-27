#!/bin/bash

# Export variable to limit OPENMP threads
export OMP_NUM_THREADS=16
# Save experiment name and 
EXPERIMENT_NAME="pretrain_on_tartanair"
export EXPERIMENT_NAME
# Save target folder to store experiment runs
cd ..
export TARGET_PATH=$(pwd)

# Go to the experiments folder
cd scripts/pre-train-and-odl/ || exit 1

# Create the folder
mkdir -p "$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_NAME/run0/"
mkdir -p "$EXPERIMENT_NAME/run1/"
mkdir -p "$EXPERIMENT_NAME/run2/"
mkdir -p "$EXPERIMENT_NAME/run3/"
mkdir -p "$EXPERIMENT_NAME/run4/"


# Run the 5 training routines
CUDA_VISIBLE_DEVICES=0 python pretrain_on_tartanair.py --tartanair_path '../../micro-tartanair/' --saved_mdl_path "$EXPERIMENT_NAME/run0/" 
# CUDA_VISIBLE_DEVICES=0 python pretrain_on_tartanair.py --tartanair_path '../../micro-tartanair/' --saved_mdl_path "$EXPERIMENT_NAME/run1/" 
# CUDA_VISIBLE_DEVICES=0 python pretrain_on_tartanair.py --tartanair_path '../../micro-tartanair/' --saved_mdl_path "$EXPERIMENT_NAME/run2/" 
# CUDA_VISIBLE_DEVICES=0 python pretrain_on_tartanair.py --tartanair_path '../../micro-tartanair/' --saved_mdl_path "$EXPERIMENT_NAME/run3/" 
# CUDA_VISIBLE_DEVICES=0 python pretrain_on_tartanair.py --tartanair_path '../../micro-tartanair/' --saved_mdl_path "$EXPERIMENT_NAME/run4/" 

# Run the 5 testing routines
CUDA_VISIBLE_DEVICES=0 python test_tartanair.py --tartanair_path '../../micro-tartanair/' --saved_mdl_path "$EXPERIMENT_NAME/run0/" --log_dir "$EXPERIMENT_NAME/run0/" --test_visual_output 'No' 
# CUDA_VISIBLE_DEVICES=0 python test_tartanair.py --tartanair_path '../../micro-tartanair/' --saved_mdl_path "$EXPERIMENT_NAME/run1/" --log_dir "$EXPERIMENT_NAME/run1/" --test_visual_output 'No' 
# CUDA_VISIBLE_DEVICES=0 python test_tartanair.py --tartanair_path '../../micro-tartanair/' --saved_mdl_path "$EXPERIMENT_NAME/run2/" --log_dir "$EXPERIMENT_NAME/run2/" --test_visual_output 'No' 
# CUDA_VISIBLE_DEVICES=0 python test_tartanair.py --tartanair_path '../../micro-tartanair/' --saved_mdl_path "$EXPERIMENT_NAME/run3/" --log_dir "$EXPERIMENT_NAME/run3/" --test_visual_output 'No' 
# CUDA_VISIBLE_DEVICES=0 python test_tartanair.py --tartanair_path '../../micro-tartanair/' --saved_mdl_path "$EXPERIMENT_NAME/run4/" --log_dir "$EXPERIMENT_NAME/run4/" --test_visual_output 'No' 


# Move the obtained folder to the main one
echo "Training completed, moving results"
mv "$EXPERIMENT_NAME" "$TARGET_PATH"
cd $TARGET_PATH
