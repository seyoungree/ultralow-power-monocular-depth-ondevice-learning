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
import utils.nyuv2_dataloader as ndataloader
# Custom loss
from utils.losses import *
# Data visualization
import matplotlib.pyplot as plt
# ODL extras
from utils.odl_utils import *


# Parser
parser = argparse.ArgumentParser("Monocular Depth Estimation CNN Test - uPyD-Net")
parser.add_argument( '--mininyuv2_path', type=str, default='../../micro-nyuv2/')
parser.add_argument( '--model_name', type=str, default='upydnet')       # 'cnn', 'upydnet' or 'upydnet_l'
parser.add_argument( '--saved_mdl_path', type=str, default='checkpoints/micro-nyuv2-8x8-label/partial_data/100perc/run0/') 
parser.add_argument( '--num_processed_images', type=int, default=6)
parser.add_argument( '--test_visual_output', type=str, default='No')   # Yes or no to test on num_processed_images the DNN qualitatively
parser.add_argument( '--save_charts', type=str, default='No')          # Save to disk the histogram with the single results on the metrics, per image
parser.add_argument( '--track_distribution', type=str, default=0)     # Save logs concerning d1, rmse, silog of every test sample (so that you can analyze the distribution) 
parser.add_argument( '--log_dir', type=str, default='./nyuv2_distr/')
parser.add_argument( '--compute_r2score', type=str, default='N')    # 'Y' to compute R2score between the dummy model and the ground truth
# Options to test the model with scale and shift related to the mean and variance of the pre-training and testing datasets (tartanair -> kitti)
parser.add_argument( '--nyuv2_dummy_predictor_path', type=str, default="./saved_models/dummy_predictors/mini-nyuv2-odl/")
parser.add_argument( '--tartanair_dummy_predictor_path', type=str, default="./saved_models/dummy_predictors/3scenarios/")
parser.add_argument( '--scale_upydnet_prediction', type=str, default='N')       # 'Y'=yes or 'N'=no. If activated, it scales the prediction of the model by the mean and variance
                                                                                # of tartanair, before, and rescales to the mean and average of mini-kitti (pixelwise stats)
parser.add_argument( '--percentage_nyuv2_testset', type=int, default=100)        # Specify the percentage of test data to be used to test (simulate lower frequency for ToF during OOD detection)
parser.add_argument( '--simulate_tof_sensor', type=str, default='N')            # 'Y'=yes or 'N'=no. If activated, ground truth is processed so to simulate the output of an 8x8 Tof sensor
parser.add_argument( '--num_tof_minima', type=int, default=1)                   # In case this is > 1, a ToF with multi-target is simulated (use this if you train the model with 'DOWNSAMPLE_PREDICTION')
args = parser.parse_args()

# Training hyperparameters and misc
MININYUV2_PATH = args.mininyuv2_path
PERCENTAGE_NYUV2_SAMPLES = args.percentage_nyuv2_testset
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
# Save the results to analyze distribution
TRACK_DISTRIBUTION = args.track_distribution
log_d1 = log_dir + '/test_d1.csv'
log_rmse = log_dir + '/test_rmse.csv'
log_silog = log_dir + '/test_silog.csv'
COMPUTE_R2SCORE = args.compute_r2score
# Scale and shift test options
NYUV2_DUMMY_PATH = args.nyuv2_dummy_predictor_path
TARTANAIR_DUMMY_PATH = args.tartanair_dummy_predictor_path
SCALE_PREDICTION = args.scale_upydnet_prediction
SIMULATE_TOF = args.simulate_tof_sensor
NUM_TOF_MINIMA = args.num_tof_minima

print("\n>>> INITIALIZING TEST INFERENCE <<<")

nyuv2_source_resolution = '48x48'
normalize_imgs          = True

testset    = ndataloader.miniNYUv2(MININYUV2_PATH, transform=False, set='test', normalize=normalize_imgs, flip_horizontally=False, test_subset_percentage=PERCENTAGE_NYUV2_SAMPLES)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

if PERCENTAGE_NYUV2_SAMPLES < 100:
    print(f"Testing on a subset of {testset.__len__()} samples ({PERCENTAGE_NYUV2_SAMPLES}% of the full mini-nyuv2 test set)")

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

# Compute the masked mean of ground truths to compute R2 Score
if COMPUTE_R2SCORE == 'Y':
    sampleGT = testset[0]['depthGT']
    avgGT = torch.zeros_like(sampleGT)
    number_occurrences_gt = torch.zeros_like(sampleGT).int()
    for test_data in testloader:
        
        # Load the proxy depth maps
        depth_gt = test_data['depthGT']
        depth_gt = depth_gt.squeeze(0)
        # Mask invalid values
        valid_mask = (depth_gt > 0)
        # Add values of the current depth map to the accumulator
        number_occurrences_gt += valid_mask
        # Sum valid values to the dummy predictor
        avgGT[valid_mask] += depth_gt[valid_mask]

    # Average values pixelwise
    avgGT = avgGT / number_occurrences_gt   



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
    kitti_mean_matrix         = torch.from_numpy(np.load(f"{NYUV2_DUMMY_PATH}/dummy_depth_predictor.npy")).to(device)
    kitti_variance_matrix     = torch.from_numpy(np.load(f"{NYUV2_DUMMY_PATH}/variance_depth_matrix.npy")).to(device)
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

# Statistics vs KITTI ground truth
test_loss = 0; test_Lap = 0; test_Lps = 0; test_La = 0; test_Lb = 0
abs_rel=0; sq_rel=0; rmse=0; rmse_log=0; d1=0; d2=0; d3=0; silog=0

# Statistics vs proxy depth maps
pr_test_loss = 0; pr_test_Lap = 0; pr_test_Lps = 0; pr_test_La = 0; pr_test_Lb = 0
pr_abs_rel=0; pr_sq_rel=0; pr_rmse=0; pr_rmse_log=0; pr_d1=0; pr_d2=0; pr_d3=0; pr_silog=0

# Statistics vs different depth ranges
# 0-15 m
test_loss_15 = 0; test_Lap_15 = 0; test_Lps_15 = 0; test_La_15 = 0; test_Lb_15 = 0
abs_rel_15=0; sq_rel_15=0; rmse_15=0; rmse_log_15=0; d1_15=0; d2_15=0; d3_15=0; silog_15=0
avg_num_15m_points = 0
# 0-25 m
test_loss_25 = 0; test_Lap_25 = 0; test_Lps_25 = 0; test_La_25 = 0; test_Lb_25 = 0
abs_rel_25=0; sq_rel_25=0; rmse_25=0; rmse_log_25=0; d1_25=0; d2_25=0; d3_25=0; silog_25=0
avg_num_25m_points = 0
# 0-50 m
test_loss_50 = 0; test_Lap_50 = 0; test_Lps_50 = 0; test_La_50 = 0; test_Lb_50 = 0
abs_rel_50=0; sq_rel_50=0; rmse_50=0; rmse_log_50=0; d1_50=0; d2_50=0; d3_50=0; silog_50=0
avg_num_50m_points = 0

if TRACK_DISTRIBUTION == True:
    with open(log_d1, 'w') as f:
        f.write(f"")
    with open(log_rmse, 'w') as f:
        f.write(f"")
    with open(log_silog, 'w') as f:
        f.write(f"")

if COMPUTE_R2SCORE == 'Y':
    # Variables for R2Score
    TSS = torch.zeros_like(avgGT).flatten().to(device)
    RSS = torch.zeros_like(avgGT).flatten().to(device)
    r2score_smpl = 0
    r2score_smpl_disp = 0
    # Plot of the sample-wise averages 
    smpl_idx = 0
    upydnet_disp_points = torch.zeros(len(testloader), 360*360)
    gt_disp_points    = torch.zeros(len(testloader), 360*360)
    upydnet_points = torch.zeros(len(testloader), 360*360)
    gt_points    = torch.zeros(len(testloader), 360*360)

with torch.no_grad():
    for test_data in testloader:

        # Get data from dictionary
        test_imgL    = test_data['imgL']
        test_depthGT = test_data['depthGT']
        test_disp    = test_data['dispL']
        test_depth   = test_data['depthL']
        test_fb      = test_data['fb']

        test_imgL = test_imgL.type(torch.FloatTensor)
        test_imgL.to(device)
        test_depthGT.to(device)
        test_disp.to(device)
        test_depth.to(device)
        test_fb.to(device)

        # calculate outputs by running images through the network
        outputs = net(test_imgL.to(device)).to(device)

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

        # In case, simulate ToF sensor
        if SIMULATE_TOF == 'Y':
            if NUM_TOF_MINIMA == 1:
                test_depth   = simulate_tof_sensor_ground_truth(test_depth, 8, 8, test_depth.size()[-2], test_depth.size()[-1], -1)
                test_depthGT = simulate_tof_sensor_ground_truth(test_depthGT.squeeze(-1), 8, 8, test_depthGT.size()[-3], test_depthGT.size()[-2], 0)
            elif NUM_TOF_MINIMA > 1:
                #test_depth_size = test_depth.size()
                test_depth   = convert_depth_label_to_multi_target_depth(test_depth.to(device), NUM_TOF_MINIMA, device, 'fp32').to(device)
                #test_depth = torch.nn.functional.intepolate(test_depth, size=(test_depth_size[-2], test_depth_size[-1]), mode='nearest')
                test_depthGT = convert_depth_label_to_multi_target_depth_GT(test_depthGT.to(device).squeeze(-1), NUM_TOF_MINIMA, device, 'fp32').to(device)

        MININYUV2_MAX_WIDTH = 360

        """
        STATISTICS WITH RESPECT TO GROUND TRUTH
        """

        # test_loss_t, test_Lap_t, test_Lps_t, test_La_t, test_Lb_t = MonocularDepthSemiSupervisedLoss(
        #                                         output         = test_outputs.to(device), 
        #                                         target         = test_disp.to(device), 
        #                                         imgL           = test_imgL.to(device), 
        #                                         imgR           = test_imgR.to(device), 
        #                                         aap            = 0.5, 
        #                                         aps            = 0.5,
        #                                         alpha          = 0.2,
        #                                         invalid_value  = -1,
        #                                         original_width = MINIKITTI_MAX_WIDTH,
        #                                         device         = device)
        # test_loss += test_loss_t;   test_loss_list.append(float(test_loss_t.to('cpu')))
        # test_Lap  += test_Lap_t
        # test_Lps  += test_Lps_t
        # test_La   += test_La_t
        # test_Lb   += test_Lb_t
    
        test_loss_t = ProxySupervisionLoss(
            output        = test_outputs.to(device),
            target        = test_disp.to(device),
            alpha         = 0.2,
            invalid_value = -1,
            device        = device
        )
        test_loss += test_loss_t;   test_loss_list.append(float(test_loss_t.to('cpu')))
        test_Lap  += 0
        test_Lps  += 0
        test_La   += 0
        test_Lb   += 0

        # Compute metrics
        if SIMULATE_TOF == 'Y' and NUM_TOF_MINIMA > 1:
            # TODO: transform test_outputs to the same shape of labels
            with torch.no_grad():
                data_transf      = PredictionMultitargetFinder(device, 'fp32')
                test_outputs_red = data_transf(test_outputs, test_depthGT)
            test_out_disp_ups  = transforms.functional.resize(img=test_outputs_red, size=[test_depthGT.size()[-2], test_depthGT.size()[-1]], interpolation=transforms.InterpolationMode.NEAREST)
            test_out_depth_ups = compute_depth_map_validation_tof(test_fb, test_out_disp_ups, device)
        elif SIMULATE_TOF == 'Y' and NUM_TOF_MINIMA == 1:
            test_out_disp_ups  = transforms.functional.resize(img=test_outputs, size=[test_depthGT.size()[-2], test_depthGT.size()[-1]], interpolation=transforms.InterpolationMode.NEAREST)
            test_out_depth_ups = compute_depth_map_validation(test_fb, test_out_disp_ups, device)            
        else: 
            test_out_disp_ups  = transforms.functional.resize(img=test_outputs, size=[test_depthGT.size()[-3], test_depthGT.size()[-2]], interpolation=transforms.InterpolationMode.NEAREST)
            test_out_depth_ups = compute_depth_map_validation(test_fb, test_out_disp_ups, device)
        test_depthGT = test_depthGT.squeeze(-1)
        run_abs_rel, run_sq_rel, run_rmse, run_rmse_log, run_d1, run_d2, run_d3 = compute_errors_masked_train(test_out_depth_ups.to(device), test_depthGT.to(device))
        run_silog = ScaleInvariantMSELogLoss(test_out_depth_ups.to(device), test_depthGT.to(device), 1.0)
        abs_rel  += run_abs_rel     ; abs_rel_list.append(float(run_abs_rel.to('cpu')))
        sq_rel   += run_sq_rel      ; sq_rel_list.append(float(run_sq_rel.to('cpu')))
        rmse     += run_rmse        ; rmse_list.append(float(run_rmse.to('cpu')))
        rmse_log += run_rmse_log    ; rmse_log_list.append(float(run_rmse_log.to('cpu')))
        d1       += run_d1          ; d1_list.append(float(run_d1.to('cpu')))
        d2       += run_d2          ; d2_list.append(float(run_d2.to('cpu')))
        d3       += run_d3          ; d3_list.append(float(run_d3.to('cpu')))
        silog    += run_silog       ; silog_list.append(float(run_silog.to('cpu')))

        if TRACK_DISTRIBUTION == True:
            with open(log_d1, 'a') as f: 
                f.write(f"{run_d1},")
            with open(log_rmse, 'a') as f: 
                f.write(f"{run_rmse},")
            with open(log_silog, 'a') as f: 
                f.write(f"{run_silog},")

        if COMPUTE_R2SCORE == 'Y':
            # # Compute part of the pixel-wise R2Score
            # dummy_out_flat = test_out_depth_ups.flatten().to(device)
            # depthGT_flat   = test_depthGT.flatten().to(device)
            # avgGT_flat     = avgGT.flatten().to(device)
            # valid_mask_r2  = torch.logical_and((depthGT_flat > 0), (avgGT_flat > 0)) 
            # TSS[valid_mask_r2] += (depthGT_flat[valid_mask_r2] - avgGT_flat[valid_mask_r2]) ** 2
            # RSS[valid_mask_r2] += (depthGT_flat[valid_mask_r2] - dummy_out_flat[valid_mask_r2]) ** 2

            # Compute the sample-wise R2Score (disparity)
            j_dummy_flat = test_out_disp_ups.flatten().to(device)
            test_disp_GT = compute_disp_map_training(test_fb, test_depthGT, device)
            j_dispGT_flat = test_disp_GT.flatten().to(device)
            vld_mask = torch.logical_and((j_dummy_flat > 0), (j_dispGT_flat > 0))
            # Create an instance of R2Score
            r2score_metric_obj = R2Score()
            r2score_metric_obj.update((j_dummy_flat[vld_mask], j_dispGT_flat[vld_mask]))
            r2score_smpl_disp += r2score_metric_obj.compute()

            # Compute the sample-wise R2Score (depth)
            i_dummy_flat = test_out_depth_ups.flatten().to(device)
            i_depthGT_flat = test_depthGT.flatten().to(device)
            vld_mask = torch.logical_and((i_dummy_flat > 0), (i_depthGT_flat > 0))
            # Create an instance of R2Score
            r2score_metric_obj = R2Score()
            r2score_metric_obj.update((i_dummy_flat[vld_mask], i_depthGT_flat[vld_mask]))
            r2score_smpl += r2score_metric_obj.compute()
            #mean_depthgt = torch.mean(i_depthGT_flat[vld_mask])
            #tss_i = torch.sum((i_depthGT_flat[vld_mask] - mean_depthgt) ** 2)
            #rss_i = torch.sum((i_depthGT_flat[vld_mask] - i_dummy_flat[vld_mask]) ** 2)
            #r2score_smpl += 1 - (rss_i / tss_i)

            # Append averages to list
            upydnet_disp_points[smpl_idx, :] = j_dummy_flat.to('cpu')
            gt_disp_points[smpl_idx, :]      = j_dispGT_flat.to('cpu')           
            upydnet_points[smpl_idx, :]      = i_dummy_flat.to('cpu')
            gt_points[smpl_idx, :]           = i_depthGT_flat.to('cpu')
            smpl_idx += 1

        """
        STATISTICS WITH RESPECT TO PROXY DEPTH MAP
        """

        # pr_test_loss_t, pr_test_Lap_t, pr_test_Lps_t, pr_test_La_t, pr_test_Lb_t = MonocularDepthSemiSupervisedLoss(
        #                                         output         = test_outputs.to(device), 
        #                                         target         = test_disp.to(device), 
        #                                         imgL           = test_imgL.to(device), 
        #                                         imgR           = test_imgR.to(device), 
        #                                         aap            = 0.5, 
        #                                         aps            = 0.5,
        #                                         alpha          = 0.2,
        #                                         invalid_value  = -1,
        #                                         original_width = MINIKITTI_MAX_WIDTH,
        #                                         device         = device)
        # pr_test_loss += pr_test_loss_t
        # pr_test_Lap  += pr_test_Lap_t
        # pr_test_Lps  += pr_test_Lps_t
        # pr_test_La   += pr_test_La_t
        # pr_test_Lb   += pr_test_Lb_t

        pr_test_loss_t = ProxySupervisionLoss(
            output        = test_outputs.to(device),
            target        = test_disp.to(device),
            alpha         = 0.2,
            invalid_value = -1,
            device        = device
        )
        pr_test_loss += pr_test_loss_t
        pr_test_Lap  += 0
        pr_test_Lps  += 0
        pr_test_La   += 0
        pr_test_Lb   += 0

        # Compute metrics
        test_out_depth = compute_depth_map_validation(test_fb, test_outputs, device)
        pr_run_abs_rel, pr_run_sq_rel, pr_run_rmse, pr_run_rmse_log, pr_run_d1, pr_run_d2, pr_run_d3 = compute_errors_masked_train(test_out_depth.to(device), test_depth.to(device))
        pr_run_silog = ScaleInvariantMSELogLoss(test_out_depth.to(device), test_depth.to(device), 1.0)
        pr_abs_rel  += pr_run_abs_rel
        pr_sq_rel   += pr_run_sq_rel
        pr_rmse     += pr_run_rmse
        pr_rmse_log += pr_run_rmse_log
        pr_d1       += pr_run_d1
        pr_d2       += pr_run_d2
        pr_d3       += pr_run_d3
        pr_silog    += pr_run_silog

            
        """
        STATISTICS WITH RESPECT TO MAX DISTANCE RANGES
        """

        max_0_15 = 1.0
        run_abs_rel_15, run_sq_rel_15, run_rmse_15, run_rmse_log_15, run_d1_15, run_d2_15, run_d3_15 = compute_errors_masked_train_max_dpth(test_out_depth_ups.to(device), test_depthGT.to(device), max_0_15)
        run_silog_15 = ScaleInvariantMSELogLoss_max_dpth(test_out_depth_ups.to(device), test_depthGT.to(device), 1.0, max_0_15)
        abs_rel_15  += run_abs_rel_15
        sq_rel_15   += run_sq_rel_15
        rmse_15     += run_rmse_15
        rmse_log_15 += run_rmse_log_15
        d1_15       += run_d1_15
        d2_15       += run_d2_15
        d3_15       += run_d3_15
        silog_15    += run_silog_15

        max_0_25 = 2.0
        run_abs_rel_25, run_sq_rel_25, run_rmse_25, run_rmse_log_25, run_d1_25, run_d2_25, run_d3_25 = compute_errors_masked_train_max_dpth(test_out_depth_ups.to(device), test_depthGT.to(device), max_0_25)
        run_silog_25 = ScaleInvariantMSELogLoss_max_dpth(test_out_depth_ups.to(device), test_depthGT.to(device), 1.0, max_0_25)
        abs_rel_25  += run_abs_rel_25
        sq_rel_25   += run_sq_rel_25
        rmse_25     += run_rmse_25
        rmse_log_25 += run_rmse_log_25
        d1_25       += run_d1_25
        d2_25       += run_d2_25
        d3_25       += run_d3_25
        silog_25    += run_silog_25

        max_0_50 = 4.0
        run_abs_rel_50, run_sq_rel_50, run_rmse_50, run_rmse_log_50, run_d1_50, run_d2_50, run_d3_50 = compute_errors_masked_train_max_dpth(test_out_depth_ups.to(device), test_depthGT.to(device), max_0_50)
        run_silog_50 = ScaleInvariantMSELogLoss_max_dpth(test_out_depth_ups.to(device), test_depthGT.to(device), 1.0, max_0_50)
        abs_rel_50  += run_abs_rel_50
        sq_rel_50   += run_sq_rel_50
        rmse_50     += run_rmse_50
        rmse_log_50 += run_rmse_log_50
        d1_50       += run_d1_50
        d2_50       += run_d2_50
        d3_50       += run_d3_50
        silog_50    += run_silog_50



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

    if COMPUTE_R2SCORE == 'Y':
        # # Compute R2Score in the end
        # R2Score_Img = torch.zeros_like(avgGT).flatten().to(device)
        # valid_mask_r2 = torch.logical_and((RSS > 0), (TSS > 0))
        # R2Score_Img[valid_mask_r2] = 1 - (RSS[valid_mask_r2] / TSS[valid_mask_r2])
        # valid_values = torch.sum(valid_mask_r2)
        # R2Score_avg = torch.sum(R2Score_Img) / valid_values
        # print(f"R2Score (avg, pixelwise)  = {R2Score_avg}")
        # Compute avg samplewise R2Score
        r2score_smpl      /= num_batches
        r2score_smpl_disp /= num_batches
        print(f"R2Score (avg, samplewise, depth) = {r2score_smpl:.3f}")
        print(f"R2Score (avg, samplewise, disparity) = {r2score_smpl_disp:.3f}")

        # Plot means to analyze R2Score (depth)
        figr2, axr2 = plt.subplots(1, 1)
        figr2.canvas.manager.set_window_title("Pixels predicted by upydnet vs ground truths (depth)")
        # Create the scatter plot
        gt_points_flat       = gt_points.flatten()
        upydnet_points_flat    = upydnet_points.flatten()
        valid_mask = torch.logical_and((gt_points_flat > 0), (upydnet_points_flat > 0))
        gt_points_flat    = gt_points_flat[valid_mask]
        upydnet_points_flat = upydnet_points_flat[valid_mask]
        axr2.scatter(gt_points_flat, upydnet_points_flat, s=3, alpha=0.1)
        # Add x and y axis labels
        axr2.set_xlabel("Ground Truth valid pixels among the test set [depth]")
        axr2.set_ylabel("uPyD-Net valid pixels among the test set [depth]")
        axr2.plot(gt_points_flat, gt_points_flat, ls="--", color='black')
        #axr2.axis("equal")
        axr2.set_xlim([0, 90])
        axr2.set_ylim([0, 90])

        # Plot means to analyze R2Score (disparity)
        figr2d, axr2d = plt.subplots(1, 1)
        figr2d.canvas.manager.set_window_title("Pixels predicted by upydnet vs ground truths (disparity)")
        # Create the scatter plot
        gt_points_disp_flat       = gt_disp_points.flatten()
        upydnet_points_disp_flat    = upydnet_disp_points.flatten()
        valid_mask = torch.logical_and((gt_points_disp_flat > 0), (upydnet_points_disp_flat > 0))
        gt_points_disp_flat    = gt_points_disp_flat[valid_mask]
        upydnet_points_disp_flat = upydnet_points_disp_flat[valid_mask]
        axr2d.scatter(gt_points_disp_flat, upydnet_points_disp_flat, s=3, alpha=0.1)
        # Add x and y axis labels
        axr2d.set_xlabel("Ground Truth valid pixels among the test set [disparity]")
        axr2d.set_ylabel("uPyD-Net valid pixels among the test set [disparity]")
        axr2d.plot(gt_points_disp_flat, gt_points_disp_flat, ls="--", color='black')
        #axr2.axis("equal")
        axr2d.set_xlim([0, 90])
        axr2d.set_ylim([0, 90])

    with open(log_file, 'w') as f:
        f.write(f"AVERAGE PREDICTION ACCURACY ON THE {size} IMAGES OF THE TEST SET WITH RESPECT TO GROUND TRUTH:\n")
        f.write(f"abs_rel = {abs_rel:.3f}, sq_rel = {sq_rel:.3f}, rmse = {rmse:.3f}, rmse_log = {rmse_log:.3f}, d1 = {d1:.3f}, d2 = {d2:.3f}, d3 = {d3:.3f}, silog = {silog:.3f}")
        f.write(f"\n\n")

    with open(log_csv, 'w') as fcsv:
        fcsv.write(f"ID,abs_rel,sq_rel,rmse,rmse_log,d1,d2,d3\n")
        fcsv.write(f"5m,{abs_rel:.3f},{sq_rel:.3f},{rmse:.3f},{rmse_log:.3f},{d1:.3f},{d2:.3f},{d3:.3f},{silog:.3f}\n")

    # """
    # STATISTICS WITH RESPECT TO PROXY DEPTH MAP
    # """

    # pr_test_loss   /= num_batches
    # pr_abs_rel     /= num_batches
    # pr_sq_rel      /= num_batches
    # pr_rmse        /= num_batches
    # pr_rmse_log    /= num_batches
    # pr_d1          /= num_batches
    # pr_d2          /= num_batches
    # pr_d3          /= num_batches
    # pr_silog       /= num_batches

    # # Print prediction statistics
    # print(f"\nAVERAGE PREDICTION ACCURACY ON THE {size} IMAGES OF THE TEST SET\nWITH RESPECT TO PROXY DEPTH MAPS:") 
    # print(f"abs_rel = {pr_abs_rel:.3f}")
    # print(f"sq_rel = {pr_sq_rel:.3f}")
    # print(f"rmse = {pr_rmse:.3f}")
    # print(f"rmse_log = {pr_rmse_log:.3f}")
    # print(f"d1 = {pr_d1:.3f}")
    # print(f"d2 = {pr_d2:.3f}")
    # print(f"d3 = {pr_d3:.3f}")
    # print(f"silog = {pr_silog:.3f}")

    # with open(log_file, 'a') as f:
    #     f.write(f"AVERAGE PREDICTION ACCURACY ON THE {size} IMAGES OF THE TEST SET WITH RESPECT TO PROXY DEPTH MAPS:\n")
    #     f.write(f"abs_rel = {pr_abs_rel:.3f}, sq_rel = {pr_sq_rel:.3f}, rmse = {pr_rmse:.3f}, rmse_log = {pr_rmse_log:.3f}, d1 = {pr_d1:.3f}, d2 = {pr_d2:.3f}, d3 = {pr_d3:.3f}, silog = {pr_silog:.3f}")
    #     f.write(f"\n\n")

    # with open(log_csv, 'a') as fcsv:
    #     fcsv.write(f"proxy_dpth,{pr_abs_rel:.3f},{pr_sq_rel:.3f},{pr_rmse:.3f},{pr_rmse_log:.3f},{pr_d1:.3f},{pr_d2:.3f},{pr_d3:.3f},{pr_silog:.3f}\n")

    # """
    # STATISTICS WITH RESPECT TO MAX DISTANCE RANGES
    # """

    # test_loss_15   /= num_batches
    # abs_rel_15     /= num_batches
    # sq_rel_15      /= num_batches
    # rmse_15        /= num_batches
    # rmse_log_15    /= num_batches
    # d1_15          /= num_batches
    # d2_15          /= num_batches
    # d3_15          /= num_batches
    # silog_15       /= num_batches

    # # Print prediction statistics
    # print(f"\nAVERAGE PREDICTION ACCURACY ON THE {size} IMAGES OF THE TEST SET (MAX DPTH 1 m)\nWITH RESPECT TO PROXY DEPTH MAPS:") 
    # print(f"abs_rel = {abs_rel_15:.3f}")
    # print(f"sq_rel = {sq_rel_15:.3f}")
    # print(f"rmse = {rmse_15:.3f}")
    # print(f"rmse_log = {rmse_log_15:.3f}")
    # print(f"d1 = {d1_15:.3f}")
    # print(f"d2 = {d2_15:.3f}")
    # print(f"d3 = {d3_15:.3f}")
    # print(f"silog = {silog_15:.3f}")

    # with open(log_file, 'a') as f:
    #     f.write(f"AVERAGE PREDICTION ACCURACY ON THE {size} IMAGES OF THE TEST SET (MAX DPTH 1 m) WITH RESPECT TO PROXY DEPTH MAPS:\n")
    #     f.write(f"abs_rel = {abs_rel_15:.3f}, sq_rel = {sq_rel_15:.3f}, rmse = {rmse_15:.3f}, rmse_log = {rmse_log_15:.3f}, d1 = {d1_15:.3f}, d2 = {d2_15:.3f}, d3 = {d3_15:.3f}, silog = {silog_15:.3f}")
    #     f.write(f"\n\n")

    # with open(log_csv, 'a') as fcsv:
    #     fcsv.write(f"15m,{abs_rel_15:.3f},{sq_rel_15:.3f},{rmse_15:.3f},{rmse_log_15:.3f},{d1_15:.3f},{d2_15:.3f},{d3_15:.3f},{silog_15:.3f}\n")

    # test_loss_25   /= num_batches
    # abs_rel_25     /= num_batches
    # sq_rel_25      /= num_batches
    # rmse_25        /= num_batches
    # rmse_log_25    /= num_batches
    # d1_25          /= num_batches
    # d2_25          /= num_batches
    # d3_25          /= num_batches
    # silog_25       /= num_batches

    # # Print prediction statistics
    # print(f"\nAVERAGE PREDICTION ACCURACY ON THE {size} IMAGES OF THE TEST SET (MAX DPTH 2 m)\nWITH RESPECT TO PROXY DEPTH MAPS:") 
    # print(f"abs_rel = {abs_rel_25:.3f}")
    # print(f"sq_rel = {sq_rel_25:.3f}")
    # print(f"rmse = {rmse_25:.3f}")
    # print(f"rmse_log = {rmse_log_25:.3f}")
    # print(f"d1 = {d1_25:.3f}")
    # print(f"d2 = {d2_25:.3f}")
    # print(f"d3 = {d3_25:.3f}")
    # print(f"silog = {silog_25:.3f}")

    # with open(log_file, 'a') as f:
    #     f.write(f"AVERAGE PREDICTION ACCURACY ON THE {size} IMAGES OF THE TEST SET (MAX DPTH 2 m) WITH RESPECT TO PROXY DEPTH MAPS:\n")
    #     f.write(f"abs_rel = {abs_rel_25:.3f}, sq_rel = {sq_rel_25:.3f}, rmse = {rmse_25:.3f}, rmse_log = {rmse_log_25:.3f}, d1 = {d1_25:.3f}, d2 = {d2_25:.3f}, d3 = {d3_25:.3f}, silog = {silog_25:.3f}")
    #     f.write(f"\n\n")

    # with open(log_csv, 'a') as fcsv:
    #     fcsv.write(f"25m,{abs_rel_25:.3f},{sq_rel_25:.3f},{rmse_25:.3f},{rmse_log_25:.3f},{d1_25:.3f},{d2_25:.3f},{d3_25:.3f},{silog_25:.3f}\n")

    # test_loss_50   /= num_batches
    # abs_rel_50     /= num_batches
    # sq_rel_50      /= num_batches
    # rmse_50        /= num_batches
    # rmse_log_50    /= num_batches
    # d1_50          /= num_batches
    # d2_50          /= num_batches
    # d3_50          /= num_batches
    # silog_50       /= num_batches

    # # Print prediction statistics
    # print(f"\nAVERAGE PREDICTION ACCURACY ON THE {size} IMAGES OF THE TEST SET (MAX DPTH 4 m)\nWITH RESPECT TO PROXY DEPTH MAPS:") 
    # print(f"abs_rel = {abs_rel_50:.3f}")
    # print(f"sq_rel = {sq_rel_50:.3f}")
    # print(f"rmse = {rmse_50:.3f}")
    # print(f"rmse_log = {rmse_log_50:.3f}")
    # print(f"d1 = {d1_50:.3f}")
    # print(f"d2 = {d2_50:.3f}")
    # print(f"d3 = {d3_50:.3f}")
    # print(f"silog = {silog_50:.3f}")

    # with open(log_file, 'a') as f:
    #     f.write(f"AVERAGE PREDICTION ACCURACY ON THE {size} IMAGES OF THE TEST SET (MAX DPTH 4 m) WITH RESPECT TO PROXY DEPTH MAPS:\n")
    #     f.write(f"abs_rel = {abs_rel_50:.3f}, sq_rel = {sq_rel_50:.3f}, rmse = {rmse_50:.3f}, rmse_log = {rmse_log_50:.3f}, d1 = {d1_50:.3f}, d2 = {d2_50:.3f}, d3 = {d3_50:.3f}, silog = {silog_50:.3f}")
    #     f.write(f"\n")

    # with open(log_csv, 'a') as fcsv:
    #     fcsv.write(f"50m,{abs_rel_50:.3f},{sq_rel_50:.3f},{rmse_50:.3f},{rmse_log_50:.3f},{d1_50:.3f},{d2_50:.3f},{d3_50:.3f},{silog_50:.3f}\n")


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
        imgL    = torch.zeros((6, single_data['imgL'].size()[0],    single_data['imgL'].size()[1], single_data['imgL'].size()[2]))
        depthGT = torch.zeros((6, single_data['depthGT'].size()[0], single_data['depthGT'].size()[1]))
        disp    = torch.zeros((6, single_data['dispL'].size()[0],   single_data['dispL'].size()[1]  ))
        depth   = torch.zeros((6, single_data['depthL'].size()[0],  single_data['depthL'].size()[1] ))
        fb      = torch.zeros((6, 1, 1))

        #indices = [0, 900, 800, 1900, 400, 1600]  # Random images
        indices = [0, 800, 400, 694, 382, 329]  # Worst cases
        #indices = [160, 168, 455, 617, 445, 461] # Best cases

        for elem in range(6):
            single_data    = testset[indices[elem]]
            imgL[elem]     = single_data['imgL']
            depthGT[elem]  = single_data['depthGT'].squeeze(-1)
            disp[elem]     = single_data['dispL']
            depth[elem]    = single_data['depthL']
            fb[elem]       = single_data['fb']

        if SIMULATE_TOF == 'Y' and NUM_TOF_MINIMA == 1:
            disp  = simulate_tof_sensor_disparity(disp, 8, 8, 360, 360, -1, 'nearest')
            depth = simulate_tof_sensor_ground_truth(depth, 8, 8, 360, 360, -1)

        imgL = imgL.type(torch.FloatTensor)
        imgL.to(device)
        depthGT = depthGT.to(device)
        disp.to(device)
        depth.to(device)
        fb.to(device)

        # Inference 
        out = net(imgL.to(device))

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

        # Compute prediction error
        abs_rel=0; sq_rel=0; rmse=0; rmse_log=0; d1=0; d2=0; d3=0; neg_pred=0
        for smpl in range(6):
            # Save depth maps from model
            dnn_disp      = dnn_out[smpl].to(device)
            dnn_depth     = compute_depth_map_validation(fb[smpl], dnn_disp, device)
            dnn_out_depths[smpl] = dnn_depth
            # Compute metrics
            dnn_depth_ups  = transforms.functional.resize(img=dnn_depth, size=[depthGT.size()[-2], depthGT.size()[-1]], interpolation=transforms.InterpolationMode.NEAREST)
            run_abs_rel, run_sq_rel, run_rmse, run_rmse_log, run_d1, run_d2, run_d3, run_neg_pred = compute_errors_masked_inference(dnn_depth_ups.to(device), depthGT[smpl].to(device))
            run_silog = ScaleInvariantMSELogLoss(dnn_depth_ups.to(device), depthGT[smpl].unsqueeze(0).to(device), 1.0)
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
        img_size     = dnn_depth_ups.size()[0] * dnn_depth_ups.size()[1] * dnn_depth_ups.size()[2]
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
        print(f"Avg Negative pixels in predicted output = {neg_pred:.2f} ({(neg_pred/img_size):.3f}%)")

        fig, ax = plt.subplots(6, 5)

        for i in range(6):
            for axis in ax[i]:
                axis.set_xticks([]), axis.set_yticks([])
            i_imgL    = imgL[i].squeeze(0).permute(1,2,0).to('cpu')
            i_dpthGT  = depthGT[i].squeeze(0).to('cpu')
            i_dpthGT  = F.interpolate(input=i_dpthGT.unsqueeze(0), scale_factor=1, mode='linear')
            i_dpthGT  = i_dpthGT.squeeze(0)
            i_dispPR  = disp[i].squeeze(0).to('cpu')
            i_depthPR = depth[i].squeeze(0).to('cpu')
            i_out     = dnn_out[i].squeeze(0).to('cpu') 
            i_dpth    = dnn_out_depths[i].squeeze(0).to('cpu')
            i_dpth    = i_dpth.squeeze(0).to('cpu') 
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

