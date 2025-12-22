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
parser.add_argument( '--saved_mdl_path', type=str, default='checkpoints/dummy_predictors/idsiadepth/') 
parser.add_argument( '--num_processed_images', type=int, default=6)
parser.add_argument( '--test_visual_output', type=str, default='Yes')   # Yes or no to test on num_processed_images the DNN qualitatively
parser.add_argument( '--set_tof_max_depth_to_invalid', type=int, default=1) # If set to 1, sets all depth values of the tof depth map corresponding to 4 meters to invalid value (-1) 
                                                                            # This tests the accuracy of the model only on valid depths 
parser.add_argument( '--log_dir', type=str, default='./')
args = parser.parse_args()

# Training hyperparameters and misc
IDSIADEPTH_PATH = args.idsiadepth_path
dummy_predictor_path = f"./{args.saved_mdl_path}"
num_processed_images = args.num_processed_images
VISUAL_TEST = args.test_visual_output
log_dir = args.log_dir
log_file = log_dir + '/disp_log.txt'
log_csv = log_dir + '/disp_log.csv'
SET_TOF_MAX_DEPTH_TO_INVALID = args.set_tof_max_depth_to_invalid


print("\n>>> INITIALIZING TESTING <<<")

# DATALOADERS
transform_test  = False
normalize_imgs  = True

idsiadepth_source_resolution = '48x48'
normalize_imgs          = True

testset    = idataloader.miniIDSIADepth(IDSIADEPTH_PATH, transform=transform_test, set='test', normalize=normalize_imgs, flip_horizontally=False)
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
DETERMINE THE DUMMY PREDICTOR BY COMPUTING THE AVERAGE OF ALL PROXY DEPTH MAPS OF IDSIADEPTH
"""

sample = testset[0]['depth']
number_of_occurrences = torch.zeros_like(sample).int()
dummy_predictor = torch.zeros_like(sample)
variance_matrix = torch.zeros_like(sample)

print("Averaging training set..")

# Mean dummy predictor
if os.path.exists(f"{dummy_predictor_path}/dummy_depth_predictor.npy"):
    model = np.load(f"{dummy_predictor_path}/dummy_depth_predictor.npy")
    dummy_predictor = torch.from_numpy(model)
else:
    for test_data in testloader:
        
        # Load the proxy depth maps
        depth_dm = test_data['depth']
        depth_dm = depth_dm.squeeze(0)
        # Set max depth to invalid value if selected
        if SET_TOF_MAX_DEPTH_TO_INVALID == 1:
            max_mask = (depth_dm == 4.0)
            depth_dm[max_mask] = -1
        # Mask invalid values
        valid_mask = (depth_dm != -1)
        # Add values of the current depth map to the accumulator
        number_of_occurrences += valid_mask
        # Sum valid values to the dummy predictor
        dummy_predictor[valid_mask] += depth_dm[valid_mask]

    # Average values pixelwise
    dummy_predictor = dummy_predictor / number_of_occurrences
    # This is the dummy prediction :)
    model = dummy_predictor.numpy()
    np.save(f"{dummy_predictor_path}/dummy_depth_predictor.npy", model)

# Variance matrix associated to the mean predictor
if os.path.exists(f"{dummy_predictor_path}/variance_depth_matrix.npy"):
    model = np.load(f"{dummy_predictor_path}/variance_depth_matrix.npy")
    variance_matrix = torch.from_numpy(model)
else:
    for test_data in testloader:
        
        # Load the proxy depth maps
        depth_dm = test_data['depth']
        depth_dm = depth_dm.squeeze(0)
        # Set max depth to invalid value if selected
        if SET_TOF_MAX_DEPTH_TO_INVALID == 1:
            max_mask = (depth_dm == 4.0)
            depth_dm[max_mask] = -1
        # Mask invalid values
        valid_mask = (depth_dm != -1)
        # Add values of the current depth map to the accumulator
        number_of_occurrences += valid_mask
        # Add value to be averaged to the variance matrix
        variance_matrix[valid_mask] += (depth_dm[valid_mask] - dummy_predictor[valid_mask])

    # Average values pixelwise
    variance_matrix = variance_matrix / number_of_occurrences
    # This is the variance matrix :)
    model = variance_matrix.numpy()
    np.save(f"{dummy_predictor_path}/variance_depth_matrix.npy", model)


print("Evaluating the accuracy on the dataset...")



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
        test_imgL = test_imgL.float().to(device)
        test_disp = test_disp.float().to(device)
        test_depth = test_depth.float().to(device)
        test_fb = test_fb.float().to(device)

        # calculate outputs by running images through the network
        outputs = dummy_predictor.float().to(device).unsqueeze(0)

        """
        STATISTICS WITH RESPECT TO GROUND TRUTH
        """

        test_loss_t = ProxySupervisionLoss(
            output        = outputs.to(device),
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
        run_abs_rel, run_sq_rel, run_rmse, run_rmse_log, run_d1, run_d2, run_d3 = compute_errors_masked_train(test_depth.to(device), outputs.to(device))
        run_silog = ScaleInvariantMSELogLoss(test_depth.to(device), outputs.to(device), 1.0)
        abs_rel  += run_abs_rel     ; abs_rel_list.append(float(run_abs_rel.to('cpu')))
        sq_rel   += run_sq_rel      ; sq_rel_list.append(float(run_sq_rel.to('cpu')))
        rmse     += run_rmse        ; rmse_list.append(float(run_rmse.to('cpu')))
        rmse_log += run_rmse_log    ; rmse_log_list.append(float(run_rmse_log.to('cpu')))
        d1       += run_d1          ; d1_list.append(float(run_d1.to('cpu')))
        d2       += run_d2          ; d2_list.append(float(run_d2.to('cpu')))
        d3       += run_d3          ; d3_list.append(float(run_d3.to('cpu')))
        silog    += run_silog       ; silog_list.append(float(run_silog.to('cpu')))


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
        indices = [0, 100, 200, 1240, 1450, 1490] # Selected

        for elem in range(6):
            single_data    = testset[indices[elem]]
            imgL[elem]     = single_data['img']
            disp[elem]     = single_data['disp']
            depth[elem]    = single_data['depth']
            fb[elem]       = single_data['fb']

        imgL  = imgL.type(torch.FloatTensor)
        imgL  = imgL.to(device)
        disp  = disp.to(device)
        depth = depth.to(device)
        fb    = fb.to(device)

        # Inference 
        out = torch.zeros_like(disp).to(device)
        for i in range(6):
            out[i, :, :] = dummy_predictor.to(device)

        # Compute prediction error
        abs_rel=0; sq_rel=0; rmse=0; rmse_log=0; d1=0; d2=0; d3=0; neg_pred=0
        for smpl in range(6):
            # Save depth maps from model
            # dnn_disp      = dnn_out[smpl].to(device)
            dnn_depth     = out[smpl].to(device) # compute_depth_map_validation(fb[smpl], dnn_disp, device)
            dnn_depth     = dnn_depth.unsqueeze(0)
            # Compute metrics
            run_abs_rel, run_sq_rel, run_rmse, run_rmse_log, run_d1, run_d2, run_d3, run_neg_pred = compute_errors_masked_inference(dnn_depth.to(device), depth[smpl].to(device))
            run_silog = ScaleInvariantMSELogLoss(dnn_depth.to(device), depth[smpl].unsqueeze(0).to(device), 1.0)
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

        fig, ax = plt.subplots(6, 4)

        for i in range(6):
            for axis in ax[i]:
                axis.set_xticks([]), axis.set_yticks([])
            i_imgL    = imgL[i].squeeze(0).permute(1,2,0).to('cpu')
            i_depthPR = depth[i].squeeze(0).to('cpu')
            i_out     = out[i].squeeze(0).to('cpu') 
            i_disp    = compute_depth_map_validation(fb[smpl], i_out.unsqueeze(0), device).squeeze(0).to('cpu')
            ax[i][0].title.set_text("Left Raw Image")
            ax[i][1].title.set_text("Depth Map (SGM)")
            ax[i][2].title.set_text("Estimated Depth")
            ax[i][3].title.set_text("Estimated Disparity")
            ax[i][0].imshow(i_imgL)
            ax[i][1].imshow(i_depthPR, 'plasma')
            ax[i][2].imshow(i_out, 'plasma')
            ax[i][3].imshow(i_disp, 'plasma')

        plt.tight_layout()
        plt.show()

