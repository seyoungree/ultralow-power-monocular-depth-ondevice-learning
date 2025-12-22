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


# Parser
parser = argparse.ArgumentParser("Monocular Depth Estimation CNN Test - uPyD-Net")
parser.add_argument( '--idsiadepth_path', type=str, default='../../idsia-micro-mde/')
parser.add_argument( '--model_name', type=str, default='upydnet')       # 'cnn', 'upydnet' or 'upydnet_l'
parser.add_argument( '--saved_mdl_path', type=str, default='checkpoints/idsia-micro-mde/sparse_update_1ep') 
parser.add_argument( '--num_processed_images', type=int, default=6)
parser.add_argument( '--test_visual_output', type=str, default='Yes')   # Yes or no to test on num_processed_images the DNN qualitatively
parser.add_argument( '--save_charts', type=str, default='Yes')          # Save to disk the histogram with the single results on the metrics, per image
parser.add_argument( '--log_dir', type=str, default='./')
# Field of view alignment between sensor and camera
parser.add_argument( '--align_cam_tof_fov', type=int, default=0)            # If set to 1, aligns fovs of camera and tof with a cropping + rescale
parser.add_argument( '--crop_border_tof_values', type=int, default=0)       # If set to 1, crops the external bits of the tof depth map and keeps only the central 6x6 values
                                                                            # If set to 0, the depth map is cropped following gemetrical boundaries with the FOVs (#FIXME: to be implemented)
parser.add_argument( '--set_tof_max_depth_to_invalid', type=int, default=1) # If set to 1, sets all depth values of the tof depth map corresponding to 4 meters to invalid value (-1) 
                                                                            # This tests the accuracy of the model only on valid depths 
# parser.add_argument( '--cam_h_fov', type=float, default=44)
# parser.add_argument( '--cam_w_fov', type=float, default=57)
# parser.add_argument( '--tof_h_fov', type=float, default=44.5)
# parser.add_argument( '--tof_w_fov', type=float, default=44.5)
# Options to test the model with scale and shift related to the mean and variance of the pre-training and testing datasets (tartanair -> kitti)
parser.add_argument( '--idsiadepth_dummy_predictor_path', type=str, default="./checkpoints/dummy_predictors/idsia-micro-mde/")
parser.add_argument( '--tartanair_dummy_predictor_path', type=str, default="./checkpoints/dummy_predictors/micro-tartanair-pre-train/")
parser.add_argument( '--scale_upydnet_prediction', type=str, default='N')       # 'Y'=yes or 'N'=no. If activated, it scales the prediction of the model by the mean and variance
                                                                                # of tartanair, before, and rescales to the mean and average of mini-kitti (pixelwise stats)
args = parser.parse_args()

# Training hyperparameters and misc
IDSIADEPTH_PATH = args.idsiadepth_path
model_name = args.model_name
statedict_dir = args.saved_mdl_path
statedict_file = f"{statedict_dir}/{model_name}.pth"
statedict_info = f"{statedict_dir}/{model_name}.info"
num_processed_images = args.num_processed_images
VISUAL_TEST = args.test_visual_output
SAVE_HISTOGRAM = args.save_charts
log_dir = args.log_dir
log_file = log_dir + '/disp_log.txt'
log_csv = log_dir + '/disp_log.csv'
# Scale and shift test options
IDSIADEPTH_DUMMY_PATH = args.idsiadepth_dummy_predictor_path
TARTANAIR_DUMMY_PATH = args.tartanair_dummy_predictor_path
SCALE_PREDICTION = args.scale_upydnet_prediction

# FOV alignment parameters
ALIGN_CAM_TOF_FOV = args.align_cam_tof_fov
CROP_BORDER_TOF_VALUES = args.crop_border_tof_values
# CAM_H_FOV = args.cam_h_fov
# CAM_W_FOV = args.cam_w_fov
# TOF_H_FOV = args.tof_h_fov
# TOF_W_FOV = args.tof_w_fov
SET_TOF_MAX_DEPTH_TO_INVALID = args.set_tof_max_depth_to_invalid

print("\n>>> INITIALIZING TEST INFERENCE <<<")

idsiadepth_source_resolution = '48x48'
normalize_imgs          = True

testset    = idataloader.miniIDSIADepth(IDSIADEPTH_PATH, transform=False, set='test', normalize=normalize_imgs, flip_horizontally=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

# Define device, models and training methods
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")


"""
GET SIZES OF INPUT DATA
"""

img_size = testset.getimagesize()
dpt_size = testset.getdepthsize()

IM_CH_IN = img_size[0]
IM_H_IN  = img_size[1]
IM_W_IN  = img_size[2]

DPTH_CH  = 1
DPTH_H   = dpt_size[0]
DPTH_W   = dpt_size[1]


"""
MODEL DEFINITION AND INITIALIZATION
"""

if model_name == 'upydnet':

    net = uPydNet(IM_CH_IN, DPTH_CH).to(device)

    # Print model structure
    print("\nModel to be trained:")
    summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=num_processed_images)
    print(f"\nInput size: [{IM_CH_IN}, {IM_H_IN}, {IM_W_IN}]")
    print(f"Output size: [{DPTH_CH}, {DPTH_H}, {DPTH_W}]")

elif model_name == 'upydnet_l':

    net = uPydNet_L(IM_CH_IN, DPTH_CH).to(device)

    # Print model structure
    print("\nModel to be trained:")
    summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=num_processed_images)
    print(f"\nInput size: [{IM_CH_IN}, {IM_H_IN}, {IM_W_IN}]")
    print(f"Output size: [{DPTH_CH}, {DPTH_H}, {DPTH_W}]")

else:
    print('Invalid model selection!!')
    exit()



"""
LOAD TRAINED MODEL
"""

# Load statedict and test trained model
print("Loading and testing trained model...")
net.load_state_dict(torch.load(statedict_file))
net.eval()

"""
LOAD DUMMY PREDICTORS IF REQUESTED
"""
if SCALE_PREDICTION == 'Y':
    kitti_mean_matrix         = torch.from_numpy(np.load(f"{IDSIADEPTH_DUMMY_PATH}/dummy_depth_predictor.npy")).to(device)
    kitti_variance_matrix     = torch.from_numpy(np.load(f"{IDSIADEPTH_DUMMY_PATH}/variance_depth_matrix.npy")).to(device)
    tartanair_mean_matrix     = torch.from_numpy(np.load(f"{TARTANAIR_DUMMY_PATH}/dummy_depth_predictor.npy")).to(device)
    tartanair_variance_matrix = torch.from_numpy(np.load(f"{TARTANAIR_DUMMY_PATH}/variance_depth_matrix.npy")).to(device)

"""
EVALUATE THE MODEL AND EXTRACT METRICS ON THE WHOLE TEST SET
"""

total = 0
num_batches = len(testloader)
size = len(testloader.dataset)

# Track the statistics with respect to the whole test set
test_loss_list = []; #test_Lap_list = []; test_Lps_list = []; test_La_list = []; test_Lb_list = []
abs_rel_list=[]; sq_rel_list=[]; rmse_list=[]; rmse_log_list=[]; d1_list=[]; d2_list=[]; d3_list=[]; silog_list=[]

# Statistics vs IDSIA DEPTH ground truth
test_loss = 0; test_Lap = 0; test_Lps = 0; test_La = 0; test_Lb = 0
abs_rel=0; sq_rel=0; rmse=0; rmse_log=0; d1=0; d2=0; d3=0; silog=0

with torch.no_grad():
    for test_data in testloader:

        # Get data from dictionary
        test_imgL    = test_data['img']
        test_disp    = test_data['disp']
        test_depth   = test_data['depth']
        test_fb      = test_data['fb']

        test_imgL = test_imgL.type(torch.FloatTensor)
        test_imgL.to(device)
        test_disp.to(device)
        test_depth.to(device)
        test_fb.to(device)

        # calculate outputs by running images through the network
        outputs = net(test_imgL.to(device)).to(device)

        if ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 1:
            test_depth = test_depth[:, 1:7, 0:7]
            test_disp  = test_disp[:, 1:7, 0:7]

            # figx, axx = plt.subplots(3)
            # axx[0].imshow(imgL[0].float().cpu().permute(1,2,0).numpy())
            # axx[1].imshow(depth[0].float().cpu().numpy())
            # axx[2].imshow(disp[0].float().cpu().numpy())
            # plt.show()
            # import pdb; pdb.set_trace()
        elif ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 0:
            pass

        # Process output with dummy predictor if requested
        if SCALE_PREDICTION == 'Y':
            outputs = (outputs - tartanair_mean_matrix) / tartanair_variance_matrix
            outputs = outputs * kitti_variance_matrix + kitti_mean_matrix

        # Add post-processing from https://openaccess.thecvf.com/content_cvpr_2017/papers/Godard_Unsupervised_Monocular_Depth_CVPR_2017_paper.pdf
        flipped_imgL      = tfun.hflip(test_imgL)
        outputs_flipped_p = net(flipped_imgL.to(device)).to(device)
        outputs_flipped   = tfun.hflip(outputs_flipped_p)

        # Average outputs and assign borders
        border_size  = int(test_imgL.size()[-1] * 0.05)
        img_width    = test_imgL.size()[-1]
        test_outputs = (outputs + outputs_flipped) / 2
        test_outputs[:, :, 0:border_size]    = outputs_flipped[:, :, 0:border_size]
        test_outputs[:, :, (img_width-1):-1] = outputs[:, :, (img_width-1):-1]

        # Squeeze results
        test_outputs = torch.squeeze(test_outputs, 1)

        # Compute depth map to compare with ToF data and upscale ToF data to align to prediction
        test_outputs_depth = compute_depth_map_test(test_fb, test_outputs, device, 'fp32').type(torch.FloatTensor)
        test_depth = upsample_tof_data(test_depth, size=[48, 48], mode='nearest').type(torch.FloatTensor)

        IDSIADEPTH_MAX_WIDTH = 360

        # import pdb; pdb.set_trace()

        if SET_TOF_MAX_DEPTH_TO_INVALID == 1:
            vmask = (test_depth == 4.0)
            test_depth[vmask] = -1
            #test_disp[mask] = -1
            #import pdb; pdb.set_trace()

        """
        STATISTICS WITH RESPECT TO GROUND TRUTH
        """
    
        # Track if all the ToF depth is invalid, in that case do not compute metrics (get NaN)
        check_mask = (test_depth != -1)
        tof_valid_values = torch.sum(check_mask)

        if SET_TOF_MAX_DEPTH_TO_INVALID == 0 or tof_valid_values > 0:

            test_loss_t = ProxySupervisionLoss(
                output        = test_outputs_depth.to(device),
                target        = test_depth.to(device),
                alpha         = 0.2,
                invalid_value = -1,
                device        = device
            )
            test_loss += test_loss_t;   test_loss_list.append(float(test_loss_t.to('cpu')))
            # test_Lap  += 0
            # test_Lps  += 0
            # test_La   += 0
            # test_Lb   += 0

            # Compute metrics
            run_abs_rel, run_sq_rel, run_rmse, run_rmse_log, run_d1, run_d2, run_d3 = compute_errors_masked_train(test_depth.to(device), test_outputs_depth.to(device))
            run_silog = ScaleInvariantMSELogLoss(test_depth.to(device), test_outputs_depth.to(device), 1.0)
            abs_rel  += run_abs_rel     ; abs_rel_list.append(float(run_abs_rel.to('cpu')))
            sq_rel   += run_sq_rel      ; sq_rel_list.append(float(run_sq_rel.to('cpu')))
            rmse     += run_rmse        ; rmse_list.append(float(run_rmse.to('cpu')))
            rmse_log += run_rmse_log    ; rmse_log_list.append(float(run_rmse_log.to('cpu')))
            d1       += run_d1          ; d1_list.append(float(run_d1.to('cpu')))
            d2       += run_d2          ; d2_list.append(float(run_d2.to('cpu')))
            d3       += run_d3          ; d3_list.append(float(run_d3.to('cpu')))
            silog    += run_silog       ; silog_list.append(float(run_silog.to('cpu')))

        # import pdb; pdb.set_trace()

    """
    STATISTICS WITH RESPECT TO GROUND TRUTH
    """

    test_loss   /= num_batches
    abs_rel     /= num_batches
    sq_rel      /= num_batches
    rmse        /= num_batches
    rmse_log    /= num_batches
    d1          /= num_batches
    d2          /= num_batches
    d3          /= num_batches
    silog       /= num_batches

    # Print prediction statistics
    print(f"\nAVERAGE PREDICTION ACCURACY ON THE {size} IMAGES OF THE TEST SET\nWITH RESPECT TO GROUND TRUTH:") 
    print(f"abs_rel = {abs_rel:.3f}")
    print(f"sq_rel = {sq_rel:.3f}")
    print(f"rmse = {rmse:.3f}")
    print(f"rmse_log = {rmse_log:.3f}")
    print(f"d1 = {d1:.3f}")
    print(f"d2 = {d2:.3f}")
    print(f"d3 = {d3:.3f}")
    print(f"silog = {silog:.3f}")
    print(f"The berHu loss is {test_loss:.3f}")

    with open(log_file, 'w') as f:
        f.write(f"AVERAGE PREDICTION ACCURACY ON THE {size} IMAGES OF THE TEST SET WITH RESPECT TO GROUND TRUTH:\n")
        f.write(f"abs_rel = {abs_rel:.3f}, sq_rel = {sq_rel:.3f}, rmse = {rmse:.3f}, rmse_log = {rmse_log:.3f}, d1 = {d1:.3f}, d2 = {d2:.3f}, d3 = {d3:.3f}, silog = {silog:.3f}")
        f.write(f"\n\n")

    with open(log_csv, 'w') as fcsv:
        fcsv.write(f"ID,abs_rel,sq_rel,rmse,rmse_log,d1,d2,d3\n")
        fcsv.write(f"4m,{abs_rel:.3f},{sq_rel:.3f},{rmse:.3f},{rmse_log:.3f},{d1:.3f},{d2:.3f},{d3:.3f},{silog:.3f}\n")

if True:
    # Create hystogram of the errors among the test set, for each metric
    fig = plt.figure(figsize=(20,10), layout='constrained')
    fig.canvas.manager.set_window_title("Error metrics over NYUv2 test set, for each sample")
    axs = fig.subplot_mosaic([['abs_rel', 'sq_rel', 'rmse', 'rmse_log'],
                              ['d1', 'd2', 'd3', 'silogloss']])
    # Y axis: number of elements, X axis: metric
    test_elems = np.arange(0, len(abs_rel_list), 1)

    axs['abs_rel'].set_title('Abs_Rel')
    axs['abs_rel'].plot(abs_rel_list, test_elems, 'k.')
    axs['abs_rel'].set_xlabel('abs_rel')
    axs['abs_rel'].set_ylabel('Img Index (test set)')
    # Plot mean
    axs['abs_rel'].axvline(x=np.mean(np.array(abs_rel_list)), color='r', linestyle='-')

    axs['sq_rel'].set_title('Sq_Rel')
    axs['sq_rel'].plot(sq_rel_list, test_elems, 'k.')
    axs['sq_rel'].set_xlabel('sq_rel')
    axs['sq_rel'].set_ylabel('Img Index (test set)')
    # Plot mean
    axs['sq_rel'].axvline(x=np.mean(np.array(sq_rel_list)), color='r', linestyle='-')

    axs['rmse'].set_title('RMSE')
    axs['rmse'].plot(rmse_list, test_elems, 'k.')
    axs['rmse'].set_xlabel('rmse')
    axs['rmse'].set_ylabel('Img Index (test set)')
    # Plot mean
    axs['rmse'].axvline(x=np.mean(np.array(rmse_list)), color='r', linestyle='-')

    axs['rmse_log'].set_title('RMSE (log)')
    axs['rmse_log'].plot(rmse_log_list, test_elems, 'k.')
    axs['rmse_log'].set_xlabel('rmse_log')
    axs['rmse_log'].set_ylabel('Img Index (test set)')
    # Plot mean
    axs['rmse_log'].axvline(x=np.mean(np.array(rmse_log_list)), color='r', linestyle='-')

    axs['d1'].set_title('delta < 1.25')
    axs['d1'].plot(d1_list, test_elems, 'k.')
    axs['d1'].set_xlabel('delta < 1.25')
    axs['d1'].set_ylabel('Img Index (test set)')
    # Plot mean
    axs['d1'].axvline(x=np.mean(np.array(d1_list)), color='r', linestyle='-')

    axs['d2'].set_title('delta < 1.25^2')
    axs['d2'].plot(d2_list, test_elems, 'k.')
    axs['d2'].set_xlabel('delta < 1.25^2')
    axs['d2'].set_ylabel('Img Index (test set)')
    # Plot mean
    axs['d2'].axvline(x=np.mean(np.array(d2_list)), color='r', linestyle='-')

    axs['d3'].set_title('delta < 1.25^3')
    axs['d3'].plot(d2_list, test_elems, 'k.')
    axs['d3'].set_xlabel('delta < 1.25^3')
    axs['d3'].set_ylabel('Img Index (test set)')
    # Plot mean
    axs['d3'].axvline(x=np.mean(np.array(d3_list)), color='r', linestyle='-')

    axs['silogloss'].set_title('Scale-Invariant MSE Loss')
    axs['silogloss'].plot(silog_list, test_elems, 'k.')
    axs['silogloss'].set_xlabel('Scale-Invariant MSE Loss')
    axs['silogloss'].set_ylabel('Img Index (test set)')
    # Plot mean
    axs['silogloss'].axvline(x=np.mean(np.array(silog_list)), color='r', linestyle='-')

    if SAVE_HISTOGRAM == 'Yes':
        pass



if VISUAL_TEST == 'Yes':
    """
    TEST THE MODEL ON SEVERAL IMAGES
    """

    with torch.no_grad():

        single_data = testset[0]

        # Get data from dictionary
        imgL    = torch.zeros((6, single_data['img'].size()[0],    single_data['img'].size()[1], single_data['img'].size()[2]))
        disp    = torch.zeros((6, single_data['disp'].size()[0],   single_data['disp'].size()[1]  ))
        depth   = torch.zeros((6, single_data['depth'].size()[0],  single_data['depth'].size()[1] ))
        fb      = torch.zeros((6, 1, 1))

        # indices = [0, 100, 200, 300, 400, 500]  # Random incremental 
        indices = [200, 780, 1150, 2410, 2660, 1600] # Selected

        for elem in range(6):
            single_data    = testset[indices[elem]]
            imgL[elem]     = single_data['img']
            disp[elem]     = single_data['disp']
            depth[elem]    = single_data['depth']
            fb[elem]       = single_data['fb']

        imgL = imgL.type(torch.FloatTensor)
        imgL.to(device)
        disp.to(device)
        depth.to(device)
        fb.to(device)

        # Inference 
        out = net(imgL.to(device))

        if ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 1:
            depth = depth[:, 1:7, 0:7]
            disp  = disp[:, 1:7, 0:7]

            # figx, axx = plt.subplots(3)
            # axx[0].imshow(imgL[0].float().cpu().permute(1,2,0).numpy())
            # axx[1].imshow(depth[0].float().cpu().numpy())
            # axx[2].imshow(disp[0].float().cpu().numpy())
            # plt.show()
            # import pdb; pdb.set_trace()
        elif ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 0:
            pass

        # Add post-processing
        flip_imgL  = tfun.hflip(imgL)
        flip_out_p = net(flip_imgL.to(device))
        flip_out   = tfun.hflip(flip_out_p)

        # Average results
        border_size  = int(imgL.size()[-1] * 0.05)
        img_width    = imgL.size()[-1]
        dnn_out      = (out + flip_out) / 2
        dnn_out[:, :, 0:border_size]    = flip_out[:, :, 0:border_size]
        dnn_out[:, :, (img_width-1):-1] = out[:, :, (img_width-1):-1]

        # Save depths from model
        dnn_out_depths = torch.zeros_like(dnn_out)

        # Upsample ToF depth maps
        tof_depth_ups = upsample_tof_data(depth, size=[48, 48], mode='nearest').type(torch.FloatTensor)

        if SET_TOF_MAX_DEPTH_TO_INVALID == 1:
            vmask = (tof_depth_ups == 4.0)
            tof_depth_ups[vmask] = -1
            #test_disp[mask] = -1
            #import pdb; pdb.set_trace()

        # Compute prediction error
        abs_rel=0; sq_rel=0; rmse=0; rmse_log=0; d1=0; d2=0; d3=0; neg_pred=0
        for smpl in range(6):
            # Save depth maps from model
            dnn_disp      = dnn_out[smpl].to(device)
            dnn_depth     = compute_depth_map_validation(fb[smpl], dnn_disp, device)
            dnn_out_depths[smpl] = dnn_depth
            # Compute metrics
            # Track if all the ToF depth is invalid, in that case do not compute metrics (get NaN)
            i_check_mask = (tof_depth_ups[smpl] != -1)
            i_tof_valid_values = torch.sum(i_check_mask)
            if SET_TOF_MAX_DEPTH_TO_INVALID == 0 or i_tof_valid_values > 0:
                run_abs_rel, run_sq_rel, run_rmse, run_rmse_log, run_d1, run_d2, run_d3, run_neg_pred = compute_errors_masked_inference(dnn_depth.to(device), tof_depth_ups[smpl].to(device))
                run_silog = ScaleInvariantMSELogLoss(dnn_depth.to(device), tof_depth_ups[smpl].unsqueeze(0).to(device), 1.0)
                abs_rel  += run_abs_rel
                sq_rel   += run_sq_rel
                rmse     += run_rmse
                rmse_log += run_rmse_log
                d1       += run_d1
                d2       += run_d2
                d3       += run_d3
                silog    += run_silog
                neg_pred += run_neg_pred
        abs_rel     /= 6
        sq_rel      /= 6
        rmse        /= 6
        rmse_log    /= 6
        d1          /= 6
        d2          /= 6
        d3          /= 6
        silog       /= 6
        neg_pred     = float(neg_pred / 6)
        img_size     = dnn_depth.size()[0] * dnn_depth.size()[1] * dnn_depth.size()[2]
        # Print prediction statistics
        print(f"\nPREDICTION ACCURACY ON THE {6} VISUALIZED IMAGES:") 
        print(f"abs_rel = {abs_rel:.3f}")
        print(f"sq_rel = {sq_rel:.3f}")
        print(f"rmse = {rmse:.3f}")
        print(f"rmse_log = {rmse_log:.3f}")
        print(f"d1 = {d1:.3f}")
        print(f"d2 = {d2:.3f}")
        print(f"d3 = {d3:.3f}")
        print(f"silog = {silog:.3f}")
        print(f"Avg Invalid Pixels in ToF Depth (the ones set to -1) = {neg_pred:.2f} ({(neg_pred/img_size):.3f}%)")

        fig, ax = plt.subplots(6, 5)

        for i in range(6):
            for axis in ax[i]:
                axis.set_xticks([]), axis.set_yticks([])

            i_imgL    = imgL[i].squeeze(0).permute(1,2,0).to('cpu')
            i_dispPR  = disp[i].squeeze(0).to('cpu')
            i_depthPR = depth[i].squeeze(0).to('cpu')
            i_out     = dnn_out[i].squeeze(0).to('cpu') 
            i_dpth    = dnn_out_depths[i].squeeze(0).to('cpu')
            i_dpth    = i_dpth.squeeze(0).to('cpu') 

            if SET_TOF_MAX_DEPTH_TO_INVALID == 1:
                # Add nans to ToF pixels with max depth
                inv_mask = (i_depthPR == 4.0)
                i_depthPR[inv_mask] = float('nan')
                i_dispPR[inv_mask] = float('nan')

            ax[i][0].title.set_text("Left Raw Image")
            ax[i][1].title.set_text("Disparity Map")
            ax[i][2].title.set_text("Estimated Disparity")
            #ax[i][3].title.set_text("Ground Truth Depth")
            ax[i][3].title.set_text("Depth Map")
            ax[i][4].title.set_text("Estimated Depth")
            ax[i][0].imshow(i_imgL)
            ax[i][1].imshow(i_dispPR, 'plasma')
            ax[i][2].imshow(i_out, 'plasma')
            #ax[i][3].imshow(i_dpthGT, 'plasma')
            ax[i][3].imshow(i_depthPR, 'plasma')
            ax[i][4].imshow(i_dpth, 'plasma')

        plt.tight_layout()
        plt.show()

