#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Copyright (C) 2024 University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
'''

'''
Authors: Davide Nadalini (d.nadalini@unibo.it)
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as tfun
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import sys
import os
import shutil
import csv
from torchsummary import summary
from torchstat import stat
from ignite.metrics.regression import R2Score
from utils.evaluation import *
# Network definitions
from models.upydnet import *
import utils.idsia_dataloader as idataloader
# Custom loss
from utils.losses import *
# Data visualization
import matplotlib.pyplot as plt
# ODL extras
from utils.odl_utils import *


# ---------------------------------------------------------------------------
# MC Dropout helpers (same pattern as training script)
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
            out = net(img)
            out = torch.squeeze(out, 1)
            preds.append(out)
    preds       = torch.stack(preds, dim=0)   # [S, B, H, W]
    mean_pred   = preds.mean(dim=0)           # [B, H, W]
    uncertainty = preds.std(dim=0)            # [B, H, W]
    return mean_pred, uncertainty


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser("Monocular Depth Estimation CNN Test - uPyd-Net (simplified, with MC Dropout)")

# Dataset / model
parser.add_argument('--idsiadepth_path', type=str, default='../../micro_sec_mde',
                    help='Root path to IDSIA Depth dataset.')
parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'val', 'test'],
                    help='Dataset split to evaluate.')
parser.add_argument('--model_name', type=str, default='upydnet',
                    help='"upydnet" or "upydnet_l".')
parser.add_argument('--checkpoint_path', type=str, required=True,
                    help='Path to trained model weights (.pth).')

# FOV / invalid pixels
parser.add_argument('--align_cam_tof_fov', type=int, default=0)
parser.add_argument('--crop_border_tof_values', type=int, default=0)
parser.add_argument('--set_tof_max_depth_to_invalid', type=int, default=1)

# Dropout / uncertainty
parser.add_argument('--mc_dropout', type=int, default=1,
                    help='1: evaluate with MC Dropout, 0: single deterministic forward.')
parser.add_argument('--mc_dropout_p', type=float, default=0.1,
                    help='Dropout probability inside uPydNet.')
parser.add_argument('--mc_dropout_samples', type=int, default=10,
                    help='Number of stochastic passes if mc_dropout=1.')

# Logging
parser.add_argument('--log_dir', type=str, default='./',
                    help='Folder where metrics log files will be written.')

args = parser.parse_args()

IDSIADEPTH_PATH = args.idsiadepth_path
dataset_split   = args.dataset_split
model_name      = args.model_name
checkpoint_path = args.checkpoint_path

ALIGN_CAM_TOF_FOV            = args.align_cam_tof_fov
CROP_BORDER_TOF_VALUES       = args.crop_border_tof_values
SET_TOF_MAX_DEPTH_TO_INVALID = args.set_tof_max_depth_to_invalid

log_dir  = args.log_dir
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'disp_log.txt')
log_csv  = os.path.join(log_dir, 'disp_log.csv')
examples_dir = os.path.join(log_dir, "examples")
os.makedirs(examples_dir, exist_ok=True)

print(f"\n>>> INITIALIZING {dataset_split.upper()} INFERENCE (SIMPLIFIED + MC DROPOUT) <<<")

normalize_imgs = True

testset = idataloader.miniIDSIADepth(
    IDSIADEPTH_PATH,
    transform=False,
    set=dataset_split,
    normalize=normalize_imgs,
    flip_horizontally=False
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=1,          # match original test script
    shuffle=False,
    num_workers=0
)

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"Using {device} device")

# Sizes
img_size = testset.getimagesize()
dpt_size = testset.getdepthsize()

IM_CH_IN = img_size[0]
IM_H_IN  = img_size[1]
IM_W_IN  = img_size[2]

DPTH_CH  = 1
DPTH_H   = dpt_size[0]
DPTH_W   = dpt_size[1]

# ---------------------------------------------------------------------------
# Model (same as original, but with dropout_p from args)
# ---------------------------------------------------------------------------

if model_name == 'upydnet':
    net = uPydNet(IM_CH_IN, DPTH_CH, dropout_p=args.mc_dropout_p).to(device)
elif model_name == 'upydnet_l':
    net = uPydNet_L(IM_CH_IN, DPTH_CH, dropout_p=args.mc_dropout_p).to(device)
else:
    print('Invalid model selection!!')
    exit()

summary_device = 'cuda' if device == 'cuda' else 'cpu'
if device == 'mps':
    net = net.to(summary_device)

print("\nModel:")
summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=1, device=summary_device)
print(f"\nInput size:  [{IM_CH_IN}, {IM_H_IN}, {IM_W_IN}]")
print(f"Output size: [{DPTH_CH}, {DPTH_H}, {DPTH_W}]")

if device == 'mps':
    net = net.to(device)

print(f"\nLoading checkpoint from: {checkpoint_path}")
net.load_state_dict(torch.load(checkpoint_path, map_location=device))
net.eval()

mode_str = f"mcd{args.mc_dropout_p}" if args.mc_dropout else "det"
print(f"Eval mode: {mode_str}")

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

num_batches = len(testloader)
size = len(testloader.dataset)

abs_rel_list=[]; sq_rel_list=[]; rmse_list=[]; rmse_log_list=[]
d1_list=[]; d2_list=[]; d3_list=[]; silog_list=[]
test_loss_list=[]

test_loss = 0.0
abs_rel  = 0.0; sq_rel   = 0.0; rmse    = 0.0
rmse_log = 0.0; d1       = 0.0; d2      = 0.0
d3       = 0.0; silog    = 0.0
mean_unc = 0.0

print(f"\n>>> Running on /{dataset_split} split <<<")

with torch.no_grad():
    for bi, test_data in enumerate(testloader):

        test_imgL  = test_data['img']
        test_disp  = test_data['disp']
        test_depth = test_data['depth']
        test_fb    = test_data['fb']

        test_imgL  = test_imgL.type(torch.FloatTensor).to(device)
        test_disp  = test_disp.to(device)
        test_depth = test_depth.to(device)
        test_fb    = test_fb.to(device)

        # Forward (with or without MC Dropout)
        # if args.mc_dropout:
        if True: 
            # test_outputs: [B, H, W]
            test_outputs, unc = mc_dropout_inference(
                net, test_imgL, n_samples=args.mc_dropout_samples, device=device
            )
            mean_unc += unc.mean().item()
        else:
            out = net(test_imgL.to(device)).to(device)   # [B, 1, H, W]
            test_outputs = torch.squeeze(out, 1)        # [B, H, W]

        # FOV alignment
        if ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 1:
            test_depth = test_depth[:, 1:7, 0:7]
            test_disp  = test_disp[:, 1:7, 0:7]
        elif ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 0:
            pass

        # Post-processing (flip-augmentation)
        flipped_imgL      = tfun.hflip(test_imgL)
        if args.mc_dropout:
            test_outputs_flip, _ = mc_dropout_inference(
                net, flipped_imgL, n_samples=args.mc_dropout_samples, device=device
            )
        else:
            outputs_flipped_p = net(flipped_imgL.to(device)).to(device)
            test_outputs_flip = torch.squeeze(outputs_flipped_p, 1)

        test_outputs_flip = tfun.hflip(test_outputs_flip.unsqueeze(1)).squeeze(1)

        border_size  = int(test_imgL.size()[-1] * 0.05)
        img_width    = test_imgL.size()[-1]
        test_outputs_filt = (test_outputs + test_outputs_flip) / 2
        test_outputs_filt[:, 0:border_size]    = test_outputs_flip[:, 0:border_size]
        test_outputs_filt[:, (img_width-1):-1] = test_outputs[:, (img_width-1):-1]

        # Compute depth map & upsample ToF depth
        test_outputs_depth = compute_depth_map_test(
            test_fb, test_outputs_filt, device, 'fp32'
        ).type(torch.FloatTensor)
        test_depth_ups = upsample_tof_data(
            test_depth, size=[48, 48], mode='nearest'
        ).type(torch.FloatTensor)

        if SET_TOF_MAX_DEPTH_TO_INVALID == 1:
            vmask = (test_depth_ups == 4.0)
            test_depth_ups[vmask] = -1

        # Skip metrics if no valid GT pixels
        check_mask      = (test_depth_ups != -1)
        tof_valid_values = torch.sum(check_mask)
        if SET_TOF_MAX_DEPTH_TO_INVALID == 1 and tof_valid_values == 0:
            continue
        
        if bi < 5:
            # Move tensors to CPU and convert to numpy
            img_np   = test_imgL[0].detach().cpu().permute(1, 2, 0).numpy()  # [H, W, C]
            pred_np  = test_outputs_depth[0].detach().cpu().numpy()          # [H, W]
            gt_np    = test_depth_ups[0].detach().cpu().numpy()              # [H, W]

            # Optional uncertainty if using MC dropout
            # if args.mc_dropout:
            if True: 
                unc_np = unc[0].detach().cpu().numpy()  # [H, W]
            else:
                unc_np = None

            # Normalize RGB to [0,1] for plotting
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

            # Mask invalid GT values for visualization (-1)
            gt_vis = gt_np.copy()
            gt_vis[gt_vis < 0] = np.nan

            # Simple vmin/vmax for depth colormap
            vmin = np.nanmin(gt_vis) if np.isfinite(gt_vis).any() else np.nanmin(pred_np)
            vmax = np.nanmax(gt_vis) if np.isfinite(gt_vis).any() else np.nanmax(pred_np)

            # Create a figure with subplots
            fig, axes = plt.subplots(1, 4 if unc_np is not None else 3, figsize=(14, 4))

            ax = axes[0]
            ax.imshow(img_np)
            ax.set_title("RGB")
            ax.axis("off")

            ax = axes[1]
            im1 = ax.imshow(gt_vis, cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title("GT Depth")
            ax.axis("off")
            fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

            ax = axes[2]
            im2 = ax.imshow(pred_np, cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title("Predicted Depth")
            ax.axis("off")
            fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

            if unc_np is not None:
                ax = axes[3]
                im3 = ax.imshow(unc_np, cmap="viridis")
                ax.set_title("Uncertainty")
                ax.axis("off")
                fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

            fig.tight_layout()

            # Save figure
            out_path = os.path.join(examples_dir, f"example_{bi:04d}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved example visualization to: {out_path}")

        # Loss (depth space, as in original script)
        test_loss_t = ProxySupervisionLoss(
            output        = test_outputs_depth.to(device),
            target        = test_depth_ups.to(device),
            alpha         = 0.2,
            invalid_value = -1,
            device        = device
        )
        test_loss += test_loss_t
        test_loss_list.append(float(test_loss_t.to('cpu')))

        # Metrics
        run_abs_rel, run_sq_rel, run_rmse, run_rmse_log, \
        run_d1, run_d2, run_d3 = compute_errors_masked_train(
            test_depth_ups.to(device), test_outputs_depth.to(device)
        )
        run_silog = ScaleInvariantMSELogLoss(
            test_depth_ups.to(device), test_outputs_depth.to(device), 1.0
        )

        abs_rel  += run_abs_rel     ; abs_rel_list.append(float(run_abs_rel.to('cpu')))
        sq_rel   += run_sq_rel      ; sq_rel_list.append(float(run_sq_rel.to('cpu')))
        rmse     += run_rmse        ; rmse_list.append(float(run_rmse.to('cpu')))
        rmse_log += run_rmse_log    ; rmse_log_list.append(float(run_rmse_log.to('cpu')))
        d1       += run_d1          ; d1_list.append(float(run_d1.to('cpu')))
        d2       += run_d2          ; d2_list.append(float(run_d2.to('cpu')))
        d3       += run_d3          ; d3_list.append(float(run_d3.to('cpu')))
        silog    += run_silog       ; silog_list.append(float(run_silog.to('cpu')))

        if (bi + 1) % 100 == 0 or (bi + 1) == num_batches:
            print(f"[{bi + 1}/{num_batches}] "
                  f"loss={test_loss/(bi+1):.4f}")

# Aggregate
test_loss /= num_batches
abs_rel   /= num_batches
sq_rel    /= num_batches
rmse      /= num_batches
rmse_log  /= num_batches
d1        /= num_batches
d2        /= num_batches
d3        /= num_batches
silog     /= num_batches
if args.mc_dropout:
    mean_unc /= num_batches

print(f"\nAVERAGE PREDICTION ACCURACY ON THE {size} {dataset_split.upper()} IMAGES:")
print(f"abs_rel = {abs_rel:.3f}")
print(f"sq_rel  = {sq_rel:.3f}")
print(f"rmse    = {rmse:.3f}")
print(f"rmse_log= {rmse_log:.3f}")
print(f"d1      = {d1:.3f}")
print(f"d2      = {d2:.3f}")
print(f"d3      = {d3:.3f}")
print(f"silog   = {silog:.3f}")
print(f"berHu loss (ProxySupervisionLoss) = {test_loss:.3f}")
if args.mc_dropout:
    print(f"mean_uncertainty (MC)           = {mean_unc:.6f}")

with open(log_file, 'w') as f:
    f.write(f"AVERAGE PREDICTION ACCURACY ON THE {size} {dataset_split.upper()} IMAGES:\n")
    f.write(f"abs_rel = {abs_rel:.3f}, sq_rel = {sq_rel:.3f}, "
            f"rmse = {rmse:.3f}, rmse_log = {rmse_log:.3f}, "
            f"d1 = {d1:.3f}, d2 = {d2:.3f}, d3 = {d3:.3f}, silog = {silog:.3f}\n")
    if args.mc_dropout:
        f.write(f"mean_uncertainty (MC) = {mean_unc:.6f}\n")

with open(log_csv, 'w') as fcsv:
    fcsv.write("ID,abs_rel,sq_rel,rmse,rmse_log,d1,d2,d3,silog\n")
    fcsv.write(f"4m,{abs_rel:.3f},{sq_rel:.3f},{rmse:.3f},"
               f"{rmse_log:.3f},{d1:.3f},{d2:.3f},{d3:.3f},{silog:.3f}\n")
