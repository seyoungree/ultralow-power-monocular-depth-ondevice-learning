'''
Copyright (C) 2024 University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Authors: Davide Nadalini (d.nadalini@unibo.it)
'''

"""
THIS CODE IS SIMILAR TO THE ONE TO TRAIN uPyD-Net FROM SCRATCH ON LOW-RESOLUTION IDSIA DEPTH
WITH DISPARITY MAPS. HOWEVER, IT REQUIRES A PRE-TRAINED MODEL TO START (PRE-TRAIN ON
LOW-RESOLUTION TARTANAIR).

DISPARITY MAPS ARE 8x8 SINCE THE DATASET IS COLLECTED WITH ToF DEPTH ONLY.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as tfun
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import sys
import time
import os
import shutil
import csv
import datetime
from distutils.dir_util import copy_tree
from torchsummary import summary
from torchstat import stat
from utils.evaluation import *
# Network definitions
from models.upydnet import *
import utils.idsia_dataloader as idataloader
import utils.dataloader as dataloader
# Custom loss
from utils.losses import *
from utils.processing import *
import utils.dump_utils as dmp
# Utils for fine-tuning
from utils.odl_utils import *
# Data visualization
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# MC Dropout helpers  (mirrors pre-training script)
# ---------------------------------------------------------------------------

def enable_mc_dropout(model):
    """Re-enable Dropout2d layers after model.eval() so MC passes are stochastic."""
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()


def mc_dropout_inference(net, img, n_samples, device):
    """
    Run n_samples stochastic forward passes with dropout active.

    Returns:
        mean_pred   : mean over n_samples  [B, H, W]
        uncertainty : std  over n_samples  [B, H, W]
    """
    net.eval()
    enable_mc_dropout(net)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = net(img)
            output = torch.squeeze(output, 1)
            preds.append(output)
    preds       = torch.stack(preds, dim=0)   # [n_samples, B, H, W]
    mean_pred   = preds.mean(dim=0)            # [B, H, W]
    uncertainty = preds.std(dim=0)             # [B, H, W]
    return mean_pred, uncertainty


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser("Monocular Depth Estimation CNN Fine-Tuning - uPyD-Net")
# Dataset and model setup
parser.add_argument( '--idsiadepth_path', type=str, default='../../idsia-micro-mde')
parser.add_argument( '--depth_v3_highest_width_resolution', type=int, default=240)
parser.add_argument( '--model_name', type=str, default='upydnet')                     # 'cnn' or 'upydnet' or 'upydnet_l'
# Fine-tuning setup
parser.add_argument( '--data_type', type=str, default='bfloat16')                     # "fp32", "fp16", "bfloat16"
parser.add_argument( '--optimizer', type=str, default='Adam')                         # 'Adam' or 'SGD'
parser.add_argument( '--downsample_prediction_or_upsample_label', type=str, default='UPSAMPLE_LABEL')
parser.add_argument( '--upsample_strategy', type=str, default='bilinear')             # 'nearest', 'bilinear', 'sparse'
parser.add_argument( '--percentage_idsiadepth_trainset', type=int, default=100)
parser.add_argument( '--track_also_test_metrics', type=int, default=0)
# Uncertainty estimation (mirrors pre-training script)
parser.add_argument( '--probabilistic',     type=int,   default=0)    # 1 = uPydNetProb, 0 = standard
parser.add_argument( '--mc_dropout',        type=int,   default=1)    # 1 = MC Dropout enabled. SR addition. 
parser.add_argument( '--mc_dropout_p',      type=float, default=0.1)  # dropout probability
parser.add_argument( '--mc_dropout_samples',type=int,   default=10)   # number of MC inference passes
# Sparse Update Options
parser.add_argument( '--su_update_encoder', type=int, default=0)
parser.add_argument( '--su_update_dec0',    type=int, default=1)
parser.add_argument( '--su_update_dec1',    type=int, default=0)
parser.add_argument( '--su_update_dec2',    type=int, default=0)
# Training setup
parser.add_argument( '--epochs', type=int, default=120 )
parser.add_argument( '--batch_size', type=int, default=16 )
parser.add_argument( '--startup_learning_rate', type=float, default=1e-4) # 1e-3 originally
parser.add_argument( '--startup_epoch', type=int, default=10)
parser.add_argument( '--init_learning_rate', type=float, default=5e-5) #1e-4 originally 
parser.add_argument( '--scheduler_epochs_step', type=int, default=10)
parser.add_argument( '--schedule_lr', type=bool, default=False)
# Field of view alignment and invalid pixels
parser.add_argument( '--align_cam_tof_fov', type=int, default=0)
parser.add_argument( '--crop_border_tof_values', type=int, default=0)
parser.add_argument( '--set_tof_max_depth_to_invalid', type=int, default=1)
# Augmentation
parser.add_argument( '--flip_horizontally_train', type=int, default=1)
# Pre-trained model path
parser.add_argument( '--pre_trained_mdl_path', type=str, default='./ckpt_upydnet_mcd0.1_20260423-173804')
# Saving and resume setup
parser.add_argument( '--save_trained_mdl', type=bool, default=True)
parser.add_argument( '--saved_mdl_path', type=str, default='saved_models/')
parser.add_argument( '--resume_from_checkpoint', type=bool, default=False)
# Other folders (checkpoint, tensorboard) — will be suffixed with run_id
parser.add_argument( '--checkpoint_folder',         type=str, default='./finetune_ckpt')
parser.add_argument( '--running_checkpoint_folder', type=str, default='./finetune_ckpt_run')
parser.add_argument( '--tensorboard_folder',        type=str, default='./finetune_run')
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Run ID  (mirrors pre-training script)
# ---------------------------------------------------------------------------

if args.probabilistic:
    mode_str = 'prob'
elif args.mc_dropout:
    mode_str = f'mcd{args.mc_dropout_p}'
else:
    mode_str = 'det'

run_id = f"finetune_{args.model_name}_{mode_str}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
print(f"\nRun ID: {run_id}")

# ---------------------------------------------------------------------------
# Training hyperparameters and paths
# ---------------------------------------------------------------------------

IDSIADEPTH_PATH = args.idsiadepth_path
model_name = args.model_name
epochs = args.epochs
batch_size = args.batch_size
MININYUV2_MAX_WIDTH = args.depth_v3_highest_width_resolution
startup_leaning_rate = args.startup_learning_rate
startup_epoch = args.startup_epoch
initial_learning_rate = args.init_learning_rate
learning_rate = initial_learning_rate
scheduler_epochs_step = args.scheduler_epochs_step
scheduler_lr_multiplier = 0.5
hflip_train = args.flip_horizontally_train

# All outputs prefixed with finetune_ + run_id → never overwrites pre-trained files or other runs
statedict_dir             = args.saved_mdl_path
statedict_file            = f"{statedict_dir}/finetune_{model_name}_{run_id}.pth"
statedict_info            = f"{statedict_dir}/finetune_{model_name}_{run_id}.info"
checkpoint_folder         = f"{args.checkpoint_folder}_{run_id}"
running_checkpoint_folder = f"{args.running_checkpoint_folder}_{run_id}"
tensorboard_folder        = f"{args.tensorboard_folder}_{run_id}"
filename                  = f"finetune_train_log_{run_id}.txt"
out_file                  = f"finetune_validation_outputs_{run_id}.txt"

resume_from_checkpoint = args.resume_from_checkpoint
SCHEDULE_LR   = args.schedule_lr
SAVE_MODEL    = args.save_trained_mdl
DELETE_CKPT_AFTER_TRAINING = False

# Options for fine-tuning
DATA_TYPE   = args.data_type
OPTIMIZER   = args.optimizer
pre_trained_mdl_path = args.pre_trained_mdl_path
PERCENTAGE_IDSIADEPTH_SAMPLES = args.percentage_idsiadepth_trainset
UPSAMPLE_STRATEGY = args.upsample_strategy
DOWNSAMPLE_PREDICTION_OR_UPSAMPLE_LABEL = args.downsample_prediction_or_upsample_label

# FOV alignment parameters
ALIGN_CAM_TOF_FOV    = args.align_cam_tof_fov
CROP_BORDER_TOF_VALUES = args.crop_border_tof_values
SET_TOF_MAX_DEPTH_TO_INVALID = args.set_tof_max_depth_to_invalid

# Sparse update
SU_UPDATE_ENC  = args.su_update_encoder
SU_UPDATE_DEC0 = args.su_update_dec0
SU_UPDATE_DEC1 = args.su_update_dec1
SU_UPDATE_DEC2 = args.su_update_dec2

# Statistics tracking
TRACK_EPOCHWISE_TEST_ACC = args.track_also_test_metrics


print("\n>>> INITIALIZING FINE-TUNING <<<")
print(f"Mode: {mode_str}")

# Initialize tensorboard (each run gets its own folder via run_id)
writer = SummaryWriter(log_dir=tensorboard_folder)

# ---------------------------------------------------------------------------
# DATALOADERS
# ---------------------------------------------------------------------------

transform_train = True
transform_val   = False
transform_test  = False
normalize_imgs  = True
HFLIP = (hflip_train != 0)
data_loader_num_workers = 0 if sys.platform == "darwin" else 2

idsiadepth_source_resolution = '48x48'

trainset    = idataloader.miniIDSIADepth(IDSIADEPTH_PATH, transform=transform_train, set='train', normalize=normalize_imgs, flip_horizontally=HFLIP, train_subset_percentage=PERCENTAGE_IDSIADEPTH_SAMPLES)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=data_loader_num_workers)

if PERCENTAGE_IDSIADEPTH_SAMPLES < 100:
    print(f"Fine-tuning on a subset of {trainset.__len__()} samples ({PERCENTAGE_IDSIADEPTH_SAMPLES}% of the full IDSIA Depth)")

valset      = idataloader.miniIDSIADepth(IDSIADEPTH_PATH, transform=transform_val, set='val', normalize=normalize_imgs, flip_horizontally=False)
valloader   = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=data_loader_num_workers)

if TRACK_EPOCHWISE_TEST_ACC:
    print("Tracking also epochwise test accuracy..")
    testset    = idataloader.miniIDSIADepth(IDSIADEPTH_PATH, transform=transform_test, set='test', normalize=normalize_imgs, flip_horizontally=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=data_loader_num_workers)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

# ---------------------------------------------------------------------------
# GET SIZES OF INPUT DATA
# ---------------------------------------------------------------------------

img_size = trainset.getimagesize()
dpt_size = trainset.getdepthsize()

IM_CH_IN = img_size[0]
IM_H_IN  = img_size[1]
IM_W_IN  = img_size[2]

# Probabilistic mode outputs 2 channels (mu + log_var), deterministic outputs 1
DPTH_CH  = 2 if args.probabilistic else 1
DPTH_H   = dpt_size[0]
DPTH_W   = dpt_size[1]

print(f"\nTrain set size: {trainset.__len__()}")
print(f"Validation set size: {valset.__len__()}")
print(f"\nHorizontal flipping augmentation set to {HFLIP}.")

# ---------------------------------------------------------------------------
# MODEL DEFINITION AND INITIALIZATION
# ---------------------------------------------------------------------------

if model_name == 'upydnet':
    if args.probabilistic:
        net = uPydNetProb(IM_CH_IN, DPTH_CH)
    elif args.mc_dropout:
        net = uPydNet(IM_CH_IN, DPTH_CH, dropout_p=args.mc_dropout_p)
    else:
        net = uPydNet(IM_CH_IN, DPTH_CH, dropout_p=0.0)

elif model_name == 'upydnet_l':
    if args.probabilistic:
        net = uPydNetProb_L(IM_CH_IN, DPTH_CH)
    elif args.mc_dropout:
        net = uPydNet_L(IM_CH_IN, DPTH_CH, dropout_p=args.mc_dropout_p)
    else:
        net = uPydNet_L(IM_CH_IN, DPTH_CH, dropout_p=0.0)

else:
    print('Invalid model selection!!')
    exit()

# Load pre-trained model
print("Loading pre-trained model..")
checkpoint_load_device = device if device == "cuda" else "cpu"
net.load_state_dict(torch.load(f"{pre_trained_mdl_path}/{model_name}.pth", map_location=checkpoint_load_device))

print("\nModel to be fine-tuned:")
summary_device = "cuda" if device == "cuda" else "cpu"
net = net.to(summary_device)
summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=batch_size, device=summary_device)
print(f"\nInput size: [{IM_CH_IN}, {IM_H_IN}, {IM_W_IN}]")
print(f"Output size: [{DPTH_CH}, {DPTH_H}, {DPTH_W}]")

# THEN cast dtype
if DATA_TYPE == 'fp32':
    net = net.to(device)
elif DATA_TYPE == 'fp16':
    net = net.half().to(device)
elif DATA_TYPE == 'bfloat16':
    net = net.bfloat16().to(device)

# Sparse Update (only for upydnet non-probabilistic)
if not args.probabilistic:
    if SU_UPDATE_ENC == 0 or SU_UPDATE_DEC0 == 0 or SU_UPDATE_DEC1 == 0 or SU_UPDATE_DEC2 == 0:
        print("[Sparse Update] Training with Sparse Update")
        if SU_UPDATE_ENC == 0:
            print("[Sparse Update] Freezing ENCODER")
            for param in net.encoder.parameters():
                param.requires_grad = False
        if SU_UPDATE_DEC0 == 0:
            print("[Sparse Update] Freezing DECODER 0")
            for param in net.decoder0.parameters():
                param.requires_grad = False
            for param in net.ups0.parameters():
                param.requires_grad = False
        if SU_UPDATE_DEC1 == 0:
            print("[Sparse Update] Freezing DECODER 1")
            for param in net.decoder1.parameters():
                param.requires_grad = False
            for param in net.ups1.parameters():
                param.requires_grad = False
        if SU_UPDATE_DEC2 == 0:
            print("[Sparse Update] Freezing DECODER 2")
            for param in net.decoder2.parameters():
                param.requires_grad = False

# ---------------------------------------------------------------------------
# OPTIMIZER
# ---------------------------------------------------------------------------

learning_rate = startup_leaning_rate
if OPTIMIZER == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=10e-8)
elif OPTIMIZER == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

def change_lr_opt(optim, lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def get_lr_opt(optim):
    for param_group in optim.param_groups:
        print(param_group['lr'])

print(f"\nSetting optimizer to {OPTIMIZER}..")

# ---------------------------------------------------------------------------
# CHECKPOINT RESUME
# ---------------------------------------------------------------------------

starting_epoch = 0
best_perf = [100] * 5
best_perf.append(0); best_perf.append(0); best_perf.append(0)

if resume_from_checkpoint and os.path.exists(running_checkpoint_folder):
    print("Loading model from checkpoint...")
    net.load_state_dict(torch.load(f"{running_checkpoint_folder}/finetune_{model_name}.pth", map_location=device))
    info_path = f"{running_checkpoint_folder}/ckpt.info"
    if os.path.exists(info_path):
        with open(info_path) as log:
            reader = csv.reader(log, delimiter=',')
            for row in reader:
                starting_epoch = int(row[0])
                learning_rate  = float(row[1])
                best_perf[0]   = float(row[2])
                best_perf[1]   = float(row[3])
                best_perf[2]   = float(row[4])
                best_perf[3]   = float(row[5])
                best_perf[4]   = float(row[6])
                best_perf[5]   = float(row[7])
                best_perf[6]   = float(row[8])
                best_perf[7]   = float(row[9])
                break

print("\nBeginning fine-tuning procedure...")

with open(filename, 'w') as file:
    file.write("--------------------------------------------------------")
    file.write(f'\nFINE-TUNING MODEL FOR {epochs} EPOCHS WITH LEARNING RATE {learning_rate}')
    file.write(f'\nMode: {mode_str}')
    if PERCENTAGE_IDSIADEPTH_SAMPLES < 100:
        file.write(f'\nTRAINING ON A SUBSET OF {len(trainset)} SAMPLES ({PERCENTAGE_IDSIADEPTH_SAMPLES}%)')
    file.write("\n--------------------------------------------------------")

previous_val_loss = 100.0
best_epoch = 0
schedule_count = starting_epoch

# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

for epoch in range(starting_epoch, epochs, 1):

    # Learning rate scheduling
    if SCHEDULE_LR:
        if epoch < startup_epoch:
            learning_rate = startup_leaning_rate
            print(f"Setting learning rate to startup value ({learning_rate})")
            schedule_count = 0
        elif epoch == startup_epoch:
            learning_rate = initial_learning_rate
            print(f"Setting learning rate to {learning_rate}")
            schedule_count = 0
        elif schedule_count >= scheduler_epochs_step:
            learning_rate = learning_rate * scheduler_lr_multiplier
            print(f"Setting learning rate to {learning_rate}")
            schedule_count = 0
        change_lr_opt(optimizer, learning_rate)
        get_lr_opt(optimizer)
    else:
        print(f"Setting learning rate to {learning_rate}")

    """
    TRAIN MODEL FOR CURRENT EPOCH
    """

    running_loss = 0.0
    net.train()

    for i, data in enumerate(trainloader):

        writer.add_scalar("Learning_Rate/Epoch", learning_rate, epoch)

        imgL    = data['img']
        disp    = data['disp']
        depth   = data['depth']
        fb      = data['fb']

        if DATA_TYPE == 'fp32':
            imgL  = imgL.to(device=device, dtype=torch.float32)
            disp  = disp.to(device=device, dtype=torch.float32)
            depth = depth.to(device=device, dtype=torch.float32)
            fb    = fb.to(device=device, dtype=torch.float32)
        elif DATA_TYPE == 'fp16':
            imgL  = imgL.to(device=device, dtype=torch.float16)
            disp  = disp.to(device=device, dtype=torch.float16)
            depth = depth.to(device=device, dtype=torch.float16)
            fb    = fb.to(device=device, dtype=torch.float16)
        elif DATA_TYPE == 'bfloat16':
            imgL  = imgL.to(device=device, dtype=torch.bfloat16)
            disp  = disp.to(device=device, dtype=torch.bfloat16)
            depth = depth.to(device=device, dtype=torch.bfloat16)
            fb    = fb.to(device=device, dtype=torch.bfloat16)

        if torch.sum(torch.isnan(imgL)) > 0:
            print('NaN values present in input!')

        optimizer.zero_grad()

        # ---- Forward pass ----
        if args.probabilistic:
            mu, log_var = net(imgL)
            mu      = torch.squeeze(mu, 1)
            log_var = torch.squeeze(log_var, 1)
            log_var = torch.clamp(log_var, min=-10.0, max=10.0)

            if ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 1:
                depth = depth[:, 1:7, 0:7]
                disp  = disp[:, 1:7, 0:7]

            if DOWNSAMPLE_PREDICTION_OR_UPSAMPLE_LABEL == 'UPSAMPLE_LABEL':
                with torch.no_grad():
                    depth = upsample_tof_data(depth, size=[48, 48], mode=UPSAMPLE_STRATEGY)
                    disp  = upsample_tof_data(disp,  size=[48, 48], mode=UPSAMPLE_STRATEGY)

            if SET_TOF_MAX_DEPTH_TO_INVALID == 1:
                mask = (depth == 4.0)
                depth[mask] = -1
                disp[mask]  = -1

            valid_mask = (disp != -1)
            diff2      = (disp - mu) ** 2
            loss_map   = 0.5 * torch.exp(-log_var) * diff2 + 0.5 * log_var
            loss       = loss_map[valid_mask].mean()

        else:
            outputs = net(imgL).to(device)
            outputs = torch.squeeze(outputs, 1)
            mu      = outputs
            log_var = None

            if ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 1:
                depth = depth[:, 1:7, 0:7]
                disp  = disp[:, 1:7, 0:7]
            elif ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 0:
                pass

            if DOWNSAMPLE_PREDICTION_OR_UPSAMPLE_LABEL == 'UPSAMPLE_LABEL':
                with torch.no_grad():
                    depth = upsample_tof_data(depth, size=[48, 48], mode=UPSAMPLE_STRATEGY)
                    disp  = upsample_tof_data(disp,  size=[48, 48], mode=UPSAMPLE_STRATEGY)
            if DOWNSAMPLE_PREDICTION_OR_UPSAMPLE_LABEL == 'DOWNSAMPLE_PREDICTION':
                print("[DOWNSAMPLE_PREDICTION] ROUTINE NOT CHECKED!!")
                pool_ker_size   = int(48 / 8)
                pool_ker_stride = int(48 / 8)
                downsampler     = nn.MaxPool2d(pool_ker_size, pool_ker_stride).to(device)
                disp            = simulate_tof_sensor_disparity(disp, 8, 8, 8, 8, -1, 'nearest')
                red_outputs     = downsampler(outputs)

            if SET_TOF_MAX_DEPTH_TO_INVALID == 1:
                mask = (depth == 4.0)
                depth[mask] = -1
                disp[mask]  = -1

            if DOWNSAMPLE_PREDICTION_OR_UPSAMPLE_LABEL == 'DOWNSAMPLE_PREDICTION':
                loss = ProxySupervisionLoss(
                    output=red_outputs.to(device), target=disp.to(device),
                    alpha=0.2, invalid_value=-1, device=device, data_type=DATA_TYPE)
            else:
                loss = ProxySupervisionLoss(
                    output=outputs.to(device), target=disp.to(device),
                    alpha=0.2, invalid_value=-1, device=device, data_type=DATA_TYPE)

        Lap = 0; Lps = loss; La = 0; Lb = 0

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            check_idx = 10 if PERCENTAGE_IDSIADEPTH_SAMPLES < 100 else 50
            if i % check_idx == check_idx - 1:
                print(f"[Epoch:{epoch}, batch:{i + 1:5d}] train_loss = {loss:.3f}")

    if SCHEDULE_LR:
        schedule_count += 1

    # -----------------------------------------------------------------------
    # VALIDATE AFTER EVERY EPOCH
    # -----------------------------------------------------------------------

    val_loss = 0
    num_batches = len(valloader)
    abs_rel=0; sq_rel=0; rmse=0; rmse_log=0; d1=0; d2=0; d3=0
    silog = 0
    val_mean_uncertainty = 0.0

    last_val_output      = 0
    last_val_uncertainty = None

    # Keep dropout active for MC dropout, otherwise go to eval
    if not args.mc_dropout:
        net.eval()

    with torch.no_grad():
        for test_data in valloader:

            val_imgL    = test_data['img']
            val_disp    = test_data['disp']
            val_depth   = test_data['depth']
            val_fb      = test_data['fb']

            if DATA_TYPE == 'fp32':
                val_imgL  = val_imgL.to(device=device, dtype=torch.float32)
                val_disp  = val_disp.to(device=device, dtype=torch.float32)
                val_depth = val_depth.to(device=device, dtype=torch.float32)
                val_fb    = val_fb.to(device=device, dtype=torch.float32)
            elif DATA_TYPE == 'fp16':
                val_imgL  = val_imgL.to(device=device, dtype=torch.float16)
                val_disp  = val_disp.to(device=device, dtype=torch.float16)
                val_depth = val_depth.to(device=device, dtype=torch.float16)
                val_fb    = val_fb.to(device=device, dtype=torch.float16)
            elif DATA_TYPE == 'bfloat16':
                val_imgL  = val_imgL.to(device=device, dtype=torch.bfloat16)
                val_disp  = val_disp.to(device=device, dtype=torch.bfloat16)
                val_depth = val_depth.to(device=device, dtype=torch.bfloat16)
                val_fb    = val_fb.to(device=device, dtype=torch.bfloat16)

            # ---- Inference (mirrors pre-training validation) ----
            if args.probabilistic:
                val_mu, val_log_var = net(val_imgL)
                val_mu      = torch.squeeze(val_mu, 1)
                val_log_var = torch.squeeze(val_log_var, 1)
                val_log_var = torch.clamp(val_log_var, min=-10.0, max=10.0)
                val_uncertainty = torch.exp(0.5 * val_log_var)   # std from log_var

            elif args.mc_dropout:
                val_mu, val_uncertainty = mc_dropout_inference(
                    net, val_imgL, n_samples=args.mc_dropout_samples, device=device)
                val_log_var = None

            else:
                val_out     = net(val_imgL).to(device)
                val_mu      = torch.squeeze(val_out, 1)
                val_log_var = None
                val_uncertainty = None

            val_outputs = val_mu   # mean prediction [B, H, W]

            # Track mean uncertainty
            if val_uncertainty is not None:
                val_mean_uncertainty += val_uncertainty.mean().item()

            # ---- FOV crop ----
            if ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 1:
                val_depth = val_depth[:, 1:7, 0:7]
                val_disp  = val_disp[:, 1:7, 0:7]
            elif ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 0:
                pass

            # Upsample ToF labels for loss and metrics
            val_disp_ups = upsample_tof_data(val_disp, size=[48, 48], mode='nearest')

            # ---- Validation loss ----
            if args.probabilistic:
                valid_mask   = (val_disp_ups != -1)
                val_loss_map = 0.5 * torch.exp(-val_log_var) * (val_disp_ups - val_outputs) ** 2 + 0.5 * val_log_var
                val_loss_t   = val_loss_map[valid_mask].mean()
            else:
                val_loss_t = ProxySupervisionLoss(
                    output=val_outputs.to(device), target=val_disp_ups.to(device),
                    alpha=0.2, invalid_value=-1, device=device, data_type=DATA_TYPE)

            val_loss += val_loss_t.item()

            # ---- Depth metrics ----
            val_out_depth_ups = upsample_tof_data(val_depth, size=[48, 48], mode='nearest')
            val_outputs_depth = compute_depth_map_test(val_fb, val_outputs, device, DATA_TYPE)
            run_abs_rel, run_sq_rel, run_rmse, run_rmse_log, run_d1, run_d2, run_d3 = \
                compute_errors_masked_train(val_out_depth_ups.to(device), val_outputs_depth)
            run_silog = ScaleInvariantMSELogLoss(val_out_depth_ups.to(device), val_outputs_depth, 1.0)
            abs_rel  += run_abs_rel
            sq_rel   += run_sq_rel
            rmse     += run_rmse
            rmse_log += run_rmse_log
            d1       += run_d1
            d2       += run_d2
            d3       += run_d3
            silog    += run_silog

            last_val_output      = val_outputs
            last_val_uncertainty = val_uncertainty

    with torch.no_grad():
        val_loss             /= num_batches
        abs_rel              /= num_batches
        sq_rel               /= num_batches
        rmse                 /= num_batches
        rmse_log             /= num_batches
        d1                   /= num_batches
        d2                   /= num_batches
        d3                   /= num_batches
        silog                /= num_batches
        val_mean_uncertainty /= num_batches

        print(f"[Epoch {epoch}] val_loss={val_loss:.4f}  uncertainty={val_mean_uncertainty:.6f}")

        with open(filename, 'a') as file:
            file.write(f"\n>>> EPOCH {epoch} <<<\n")
            file.write(f"LR = {learning_rate}, avg_loss = {val_loss:>8f}\n")
            file.write(f"abs_rel          = {abs_rel:.3f}\n")
            file.write(f"sq_rel           = {sq_rel:.3f}\n")
            file.write(f"rmse             = {rmse:.3f}\n")
            file.write(f"rmse_log         = {rmse_log:.3f}\n")
            file.write(f"d1               = {d1:.3f}\n")
            file.write(f"d2               = {d2:.3f}\n")
            file.write(f"d3               = {d3:.3f}\n")
            file.write(f"silogloss        = {silog:.3f}\n")
            file.write(f"mean_uncertainty = {val_mean_uncertainty:.6f}  (mode={mode_str})\n")

        with open(out_file, 'w') as file:
            file.write(f"\n>>> Epoch {epoch}, Validation output:\n")
            file.write(f"{dmp.tensor_to_string(last_val_output)}")

        """
        TRACK LOSS AND QUALITY METRICS AFTER EACH EPOCH
        """

        writer.add_scalar("Train_Loss/Epoch",      loss,                 epoch)
        writer.add_scalar("Val_Loss/Epoch",        val_loss,             epoch)
        writer.add_scalar("Abs_rel/Epoch",         abs_rel,              epoch)
        writer.add_scalar("Sq_rel/Epoch",          sq_rel,               epoch)
        writer.add_scalar("RMSE/Epoch",            rmse,                 epoch)
        writer.add_scalar("RMSE_log/Epoch",        rmse_log,             epoch)
        writer.add_scalar("d1/Epoch",              d1,                   epoch)
        writer.add_scalar("d2/Epoch",              d2,                   epoch)
        writer.add_scalar("d3/Epoch",              d3,                   epoch)
        writer.add_scalar("silogloss/Epoch",       silog,                epoch)
        writer.add_scalar("Val_Uncertainty/Epoch", val_mean_uncertainty, epoch)

        # Probabilistic-specific tracking (mirrors pre-training)
        if args.probabilistic and last_val_uncertainty is not None:
            writer.add_scalar("Val_LogVar/Mean",    val_log_var.mean(), epoch)
            writer.add_scalar("Val_LogVar/Std",     val_log_var.std(),  epoch)
            writer.add_scalar("Val_LogVar/Min",     val_log_var.min(),  epoch)
            writer.add_scalar("Val_LogVar/Max",     val_log_var.max(),  epoch)
            writer.add_scalar("Val_Variance/Mean",  torch.exp(val_log_var).mean(), epoch)

        if TRACK_EPOCHWISE_TEST_ACC == 0:
            writer.flush()

        """
        IF SELECTED, PERFORM A TEST IN EACH EPOCH TO TRACK EVOLUTION
        """
        if TRACK_EPOCHWISE_TEST_ACC == 1:

            print(f"Testing epoch {epoch} (mode={mode_str})..")

            num_test_batches = len(testloader)

            test_loss     = 0
            test_abs_rel  = 0; test_sq_rel   = 0; test_rmse    = 0
            test_rmse_log = 0; test_d1        = 0; test_d2      = 0
            test_d3       = 0; test_silog     = 0
            test_mean_uncertainty = 0.0

            test_loss_list=[]; test_abs_rel_list=[]; test_sq_rel_list=[]
            test_rmse_list=[]; test_rmse_log_list=[]; test_d1_list=[]
            test_d2_list=[]; test_d3_list=[]; test_silog_list=[]

            with torch.no_grad():
                for test_data in testloader:

                    test_imgL  = test_data['img']
                    test_disp  = test_data['disp']
                    test_depth = test_data['depth']
                    test_fb    = test_data['fb']

                    if DATA_TYPE == 'fp32':
                        test_imgL  = test_imgL.to(device=device, dtype=torch.float32)
                        test_disp  = test_disp.to(device=device, dtype=torch.float32)
                        test_depth = test_depth.to(device=device, dtype=torch.float32)
                        test_fb    = test_fb.to(device=device, dtype=torch.float32)
                    elif DATA_TYPE == 'fp16':
                        test_imgL  = test_imgL.to(device=device, dtype=torch.float16)
                        test_disp  = test_disp.to(device=device, dtype=torch.float16)
                        test_depth = test_depth.to(device=device, dtype=torch.float16)
                        test_fb    = test_fb.to(device=device, dtype=torch.float16)
                    elif DATA_TYPE == 'bfloat16':
                        test_imgL  = test_imgL.to(device=device, dtype=torch.bfloat16)
                        test_disp  = test_disp.to(device=device, dtype=torch.bfloat16)
                        test_depth = test_depth.to(device=device, dtype=torch.bfloat16)
                        test_fb    = test_fb.to(device=device, dtype=torch.bfloat16)

                    # ---- Inference ----
                    if args.probabilistic:
                        test_mu, test_log_var = net(test_imgL)
                        test_mu      = torch.squeeze(test_mu, 1)
                        test_log_var = torch.squeeze(test_log_var, 1)
                        test_log_var = torch.clamp(test_log_var, min=-10.0, max=10.0)
                        test_uncertainty = torch.exp(0.5 * test_log_var)
                    elif args.mc_dropout:
                        test_mu, test_uncertainty = mc_dropout_inference(
                            net, test_imgL, n_samples=args.mc_dropout_samples, device=device)
                        test_log_var = None
                    else:
                        test_out     = net(test_imgL).to(device)
                        test_mu      = torch.squeeze(test_out, 1)
                        test_log_var = None
                        test_uncertainty = None

                    test_outputs = test_mu
                    if test_uncertainty is not None:
                        test_mean_uncertainty += test_uncertainty.mean().item()

                    if ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 1:
                        test_depth = test_depth[:, 1:7, 0:7]
                        test_disp  = test_disp[:, 1:7, 0:7]
                    elif ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 0:
                        pass

                    # Post-processing: average with horizontally flipped output
                    flipped_imgL = tfun.hflip(test_imgL)
                    if args.probabilistic:
                        test_mu_flip, _ = net(flipped_imgL.to(device))
                        test_mu_flip    = torch.squeeze(test_mu_flip, 1)
                    elif args.mc_dropout:
                        test_mu_flip, _ = mc_dropout_inference(
                            net, flipped_imgL, n_samples=args.mc_dropout_samples, device=device)
                    else:
                        test_mu_flip    = torch.squeeze(net(flipped_imgL.to(device)).to(device), 1)

                    test_outputs_flipped = tfun.hflip(test_mu_flip.unsqueeze(1)).squeeze(1)

                    border_size  = int(test_imgL.size()[-1] * 0.05)
                    img_width    = test_imgL.size()[-1]
                    test_outputs_filt = (test_outputs + test_outputs_flipped) / 2
                    test_outputs_filt[:, 0:border_size]    = test_outputs_flipped[:, 0:border_size]
                    test_outputs_filt[:, (img_width-1):-1] = test_outputs[:, (img_width-1):-1]

                    test_outputs_depth = compute_depth_map_test(test_fb, test_outputs_filt, device, 'fp32').type(torch.FloatTensor)
                    test_depth_ups     = upsample_tof_data(test_depth, size=[48, 48], mode='nearest').type(torch.FloatTensor)

                    if SET_TOF_MAX_DEPTH_TO_INVALID == 1:
                        vmask = (test_depth_ups == 4.0)
                        test_depth_ups[vmask] = -1

                    test_loss_t = Test_ProxySupervisionLoss(
                        output=test_outputs_depth.to(device), target=test_depth_ups.to(device),
                        alpha=0.2, invalid_value=-1, device=device, data_type=DATA_TYPE)
                    test_loss += test_loss_t.item()
                    test_loss_list.append(float(test_loss_t.to('cpu')))

                    run_abs_rel, run_sq_rel, run_rmse, run_rmse_log, run_d1, run_d2, run_d3 = \
                        compute_errors_masked_train(test_depth_ups.to(device), test_outputs_depth.to(device))
                    run_silog = ScaleInvariantMSELogLoss(test_depth_ups.to(device), test_outputs_depth.to(device), 1.0)

                    test_abs_rel  += run_abs_rel  ; test_abs_rel_list.append(float(run_abs_rel.to('cpu')))
                    test_sq_rel   += run_sq_rel   ; test_sq_rel_list.append(float(run_sq_rel.to('cpu')))
                    test_rmse     += run_rmse     ; test_rmse_list.append(float(run_rmse.to('cpu')))
                    test_rmse_log += run_rmse_log ; test_rmse_log_list.append(float(run_rmse_log.to('cpu')))
                    test_d1       += run_d1       ; test_d1_list.append(float(run_d1.to('cpu')))
                    test_d2       += run_d2       ; test_d2_list.append(float(run_d2.to('cpu')))
                    test_d3       += run_d3       ; test_d3_list.append(float(run_d3.to('cpu')))
                    test_silog    += run_silog    ; test_silog_list.append(float(run_silog.to('cpu')))

                test_loss             /= num_test_batches
                test_abs_rel          /= num_test_batches
                test_sq_rel           /= num_test_batches
                test_rmse             /= num_test_batches
                test_rmse_log         /= num_test_batches
                test_d1               /= num_test_batches
                test_d2               /= num_test_batches
                test_d3               /= num_test_batches
                test_silog            /= num_test_batches
                test_mean_uncertainty /= num_test_batches

                writer.add_scalar("Test_Loss/Epoch",            test_loss,             epoch)
                writer.add_scalar("Test_Abs_rel/Epoch",         test_abs_rel,          epoch)
                writer.add_scalar("Test_Sq_rel/Epoch",          test_sq_rel,           epoch)
                writer.add_scalar("Test_RMSE/Epoch",            test_rmse,             epoch)
                writer.add_scalar("Test_RMSE_log/Epoch",        test_rmse_log,         epoch)
                writer.add_scalar("Test_d1/Epoch",              test_d1,               epoch)
                writer.add_scalar("Test_d2/Epoch",              test_d2,               epoch)
                writer.add_scalar("Test_d3/Epoch",              test_d3,               epoch)
                writer.add_scalar("Test_silogloss/Epoch",       test_silog,            epoch)
                writer.add_scalar("Test_Uncertainty/Epoch",     test_mean_uncertainty, epoch)
                writer.flush()

                print(f"Testing complete! uncertainty={test_mean_uncertainty:.6f}")

        """
        CHECKPOINT MODEL AND CHECK IF THE BEST ACCURACY IS REACHED
        """

        checkpoint_model = False
        if (val_loss >= 0) and (val_loss < previous_val_loss):
            checkpoint_model  = True
            best_perf         = [val_loss, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3]
            best_epoch        = epoch
            previous_val_loss = val_loss

        # Best ckpt block
        if checkpoint_model:
            os.makedirs(checkpoint_folder, exist_ok=True)
            with open(f"{checkpoint_folder}/model.info", 'w') as log:
                log.write(f"--- ckpt.info organization (tracks best epoch): ---\n")
                log.write(f"epoch,learning_rate,test_loss,abs_rel,sq_rel,rmse,rmse_log,d1,d2,d3\n")
                # summary removed — architecture printed once at init
            with open(f"{checkpoint_folder}/ckpt.info", 'w') as log:
                log.write(f"{best_epoch},{learning_rate},")
                log.write(f"{best_perf[0]},{best_perf[1]},{best_perf[2]},{best_perf[3]},{best_perf[4]},{best_perf[5]},{best_perf[6]},{best_perf[7]}\n")
            torch.save(net.state_dict(), f"{checkpoint_folder}/finetune_{model_name}.pth")

        # Running ckpt block
        os.makedirs(running_checkpoint_folder, exist_ok=True)
        with open(f"{running_checkpoint_folder}/model.info", 'w') as log:
            log.write(f"--- ckpt.info organization (tracks last training epoch): ---\n")
            log.write(f"epoch,learning_rate,test_loss,abs_rel,sq_rel,rmse,rmse_log,d1,d2,d3\n")
            # summary removed — architecture printed once at init
        with open(f"{running_checkpoint_folder}/ckpt.info", 'w') as log:
            log.write(f"{epoch},{learning_rate},")
            log.write(f"{val_loss},{abs_rel},{sq_rel},{rmse},{rmse_log},{d1},{d2},{d3}\n")
        torch.save(net.state_dict(), f"{running_checkpoint_folder}/finetune_{model_name}.pth")

        # Final model save block
        os.makedirs(statedict_dir, exist_ok=True)
        with open(statedict_info, 'w') as f:
            f.write("--- Input and label sizes: ---\n")
            f.write(f"Input size: [{IM_CH_IN}, {IM_H_IN}, {IM_W_IN}]\n")
            f.write(f"Label size: [{DPTH_CH}, {DPTH_H}, {DPTH_W}]\n")
            f.write("\n--- Hyperparameters: ---\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Learning rate: {initial_learning_rate} to {learning_rate} (scale every {scheduler_epochs_step} epochs)\n")
            f.write(f"Mode: {mode_str}\n")
            if args.mc_dropout:
                f.write(f"MC Dropout p: {args.mc_dropout_p}, samples: {args.mc_dropout_samples}\n")
            f.write("\n--- Results: ---\n")
            f.write(f"best_epoch  = {best_epoch:.3f}\n")
            f.write(f"abs_rel     = {best_perf[1]:.3f}\n")
            f.write(f"sq_rel      = {best_perf[2]:.3f}\n")
            f.write(f"rmse        = {best_perf[3]:.3f}\n")
            f.write(f"rmse_log    = {best_perf[4]:.3f}\n")
            f.write(f"d1          = {best_perf[5]:.3f}\n")
            f.write(f"d2          = {best_perf[6]:.3f}\n")
            f.write(f"d3          = {best_perf[7]:.3f}\n")
            # summary removed — architecture printed once at init

# ---------------------------------------------------------------------------
# SAVE FINAL MODEL
# ---------------------------------------------------------------------------

net.eval()

with torch.no_grad():

    print("Finished fine-tuning!")

    if SAVE_MODEL:
        print(f"Saving fine-tuned model weights to {statedict_file}..")
        os.makedirs(statedict_dir, exist_ok=True)
        if os.path.exists(checkpoint_folder):
            shutil.copyfile(f"{checkpoint_folder}/finetune_{model_name}.pth", statedict_file)
            copy_tree(f"{tensorboard_folder}/", f"{statedict_dir}/finetune_tensorboard_data_{run_id}/")
        else:
            torch.save(net.state_dict(), statedict_file)
            copy_tree(f"{tensorboard_folder}/", f"{statedict_dir}/finetune_tensorboard_data_{run_id}/")
        with open(statedict_info, 'w') as f:
            f.write("--- Input and label sizes: ---\n")
            f.write(f"Input size: [{IM_CH_IN}, {IM_H_IN}, {IM_W_IN}]\n")
            f.write(f"Label size: [{DPTH_CH}, {DPTH_H}, {DPTH_W}]\n")
            f.write("\n--- Hyperparameters: ---\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Learning rate: {initial_learning_rate} to {learning_rate} (scale every {scheduler_epochs_step} epochs)\n")
            f.write(f"Mode: {mode_str}\n")
            if args.mc_dropout:
                f.write(f"MC Dropout p: {args.mc_dropout_p}, samples: {args.mc_dropout_samples}\n")
            f.write("\n--- Results: ---\n")
            f.write(f"best_epoch  = {best_epoch:.3f}\n")
            f.write(f"abs_rel     = {best_perf[1]:.3f}\n")
            f.write(f"sq_rel      = {best_perf[2]:.3f}\n")
            f.write(f"rmse        = {best_perf[3]:.3f}\n")
            f.write(f"rmse_log    = {best_perf[4]:.3f}\n")
            f.write(f"d1          = {best_perf[5]:.3f}\n")
            f.write(f"d2          = {best_perf[6]:.3f}\n")
            f.write(f"d3          = {best_perf[7]:.3f}\n")
            f.write("\n--- Trained Model: ---\n\n")

    if DELETE_CKPT_AFTER_TRAINING:
        print("Removing checkpoint folder after training..")
        shutil.rmtree(checkpoint_folder)

    writer.close()
