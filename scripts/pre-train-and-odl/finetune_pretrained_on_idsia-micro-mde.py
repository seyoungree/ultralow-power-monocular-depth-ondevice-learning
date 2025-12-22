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

# Parser
parser = argparse.ArgumentParser("Monocular Depth Estimation CNN Fine-Tuning - uPyD-Net")
# Dataset and model setup
parser.add_argument( '--idsiadepth_path', type=str, default='../../idsia-micro-mde/')
parser.add_argument( '--depth_v3_highest_width_resolution', type=int, default=240)  # Needed to compute the rescaling of disparity maps in image reprojection loss
parser.add_argument( '--model_name', type=str, default='upydnet')                     # 'cnn' or 'upydnet' or 'upydnet_l'
# Fine-tuning setup
parser.add_argument( '--data_type', type=str, default='bfloat16')                         # Sets the data type for training: "fp32", "fp16", "bfloat16"
parser.add_argument( '--optimizer', type=str, default='Adam')                         # 'Adam' or 'SGD'. The latter is more compatible with On-Device Learning memory requirements
parser.add_argument( '--downsample_prediction_or_upsample_label', type=str, default='UPSAMPLE_LABEL')
                                                                                      # IMPORTANT: --downsample_prediction_or_upsample_label selects the strategy for ToF training (selecting also proxy_disparity_resolution = "8x8")
                                                                                      #            'UPSAMPLE_LABEL': upsamples each 8x8 label to a 48x48 label using the strategy defined with '--upsample_strategy'
                                                                                      #            'DOWNSAMPLE_PREDICTION': instead of upsampling the label, downsamples the prediction to match a Nx8x8 label provided by a simulated ToF sensor, 
                                                                                      #                          providing N minima (1 to 4), distant at least 600 mm one to the next, then trains with this transformation.
parser.add_argument( '--upsample_strategy', type=str, default='bilinear')              # If proxy_disparity_resolution is "8x8", the disparity maps need to be upsampled for training ('UPSAMPLE_LABEL'). 
                                                                                      # You can choose between: "nearest": the value of each pixel of the 8x8 label is copied into each corresponding 6x6 field in the 48x48 upscaled label
                                                                                      #                         "bilinear": the values of the 8x8 label are interpolated to create the 48x48 upscaled label
                                                                                      #                         "sparse": the 48x48 upscaled label is invalid in all points of each 6x6 field, excluded the one corresponding to the valid value of the 8x8 disparity map
parser.add_argument( '--percentage_idsiadepth_trainset', type=int, default=100)             # Specify the percentage of training data in the NYUV2 train set to be used in training
parser.add_argument( '--track_also_test_metrics', type=int, default=0)                # If set to 1, on each epoch a test is made, and metrics are tracked in the tensorboard (to see the accuracy sweep)
# Sparse Update Options
parser.add_argument( '--su_update_encoder', type=int, default=1)                      # Set to 0 if you don't want to update Encoder 
parser.add_argument( '--su_update_dec0',    type=int, default=1)                      # Set to 0 if you don't want to update Decoder 0
parser.add_argument( '--su_update_dec1',    type=int, default=1)                      # Set to 0 if you don't want to update Decoder 1
parser.add_argument( '--su_update_dec2',    type=int, default=1)                      # Set to 0 if you don't want to update Decoder 2
# Training setup
parser.add_argument( '--epochs', type=int, default=1 )
parser.add_argument( '--batch_size', type=int, default=16 )
parser.add_argument( '--startup_learning_rate', type=float, default=1e-3)    # First epoch's learning rate to avoid gradient explosion
parser.add_argument( '--startup_epoch', type=int, default=10)                 # Epoch after which to apply the init_learning_rate
parser.add_argument( '--init_learning_rate', type=float, default=1e-4)        # Initial learning rate after first startup epochs
parser.add_argument( '--scheduler_epochs_step', type=int, default=10)
parser.add_argument( '--schedule_lr', type=bool, default=False)
# Field of view alignment between sensor and camera adn label invalid pixels
parser.add_argument( '--align_cam_tof_fov', type=int, default=0)            # If set to 1, aligns fovs of camera and tof with a cropping + rescale
parser.add_argument( '--crop_border_tof_values', type=int, default=0)       # If set to 1, crops the external bits of the tof depth map and keeps only the central 6x6 values
                                                                            # If set to 0, the depth map is cropped following gemetrical boundaries with the FOVs (#FIXME: to be implemented)
parser.add_argument( '--set_tof_max_depth_to_invalid', type=int, default=1) # If set to 1, sets all depth values of the tof depth map corresponding to 4 meters to invalid value (-1) 
# parser.add_argument( '--cam_h_fov', type=float, default=44)
# parser.add_argument( '--cam_w_fov', type=float, default=57)
# parser.add_argument( '--tof_h_fov', type=float, default=44.5)
# parser.add_argument( '--tof_w_fov', type=float, default=44.5)
# Augmentation
parser.add_argument( '--flip_horizontally_train', type=int, default=1)          # Select if to flip or not the images and labels during training
# Loss tuning parameters
parser.add_argument( '--pre_trained_mdl_path', type=str, default='./checkpoints/micro-tartanair-pre-train/')   # Location of the pre-trained model on tartanair
# Saving and resume setup
parser.add_argument( '--save_trained_mdl', type=bool, default=True)
parser.add_argument( '--saved_mdl_path', type=str, default='saved_models/')  # Where to save the model
parser.add_argument( '--resume_from_checkpoint', type=bool, default=False)
# Other folders (checkpoint, tensorboard)
parser.add_argument( '--checkpoint_folder', type=str, default='./ckpt')                 # Folder for the best performing model
parser.add_argument( '--running_checkpoint_folder', type=str, default='./ckpt_run')     # Folder with last epoch's model
parser.add_argument( '--tensorboard_folder', type=str, default='./run')                 # Folder to store tensorboard data about training
args = parser.parse_args()

# Training hyperparameters and misc
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
filename = "train_log.txt"
out_file = "validation_outputs.txt"
statedict_dir = args.saved_mdl_path
statedict_file = f"{statedict_dir}/{model_name}.pth"
statedict_info = f"{statedict_dir}/{model_name}.info"
resume_from_checkpoint = args.resume_from_checkpoint
checkpoint_folder = args.checkpoint_folder                        
running_checkpoint_folder = args.running_checkpoint_folder              
tensorboard_folder = args.tensorboard_folder
SCHEDULE_LR = args.schedule_lr
SAVE_MODEL = args.save_trained_mdl
# Set to True if you want to delete ckpt after training
DELETE_CKPT_AFTER_TRAINING = False

# Options for fine-tuning
DATA_TYPE = args.data_type
OPTIMIZER = args.optimizer
pre_trained_mdl_path = args.pre_trained_mdl_path
PERCENTAGE_IDSIADEPTH_SAMPLES = args.percentage_idsiadepth_trainset
UPSAMPLE_STRATEGY = args.upsample_strategy
DOWNSAMPLE_PREDICTION_OR_UPSAMPLE_LABEL = args.downsample_prediction_or_upsample_label

# FOV alignment parameters
ALIGN_CAM_TOF_FOV = args.align_cam_tof_fov
CROP_BORDER_TOF_VALUES = args.crop_border_tof_values
# CAM_H_FOV = args.cam_h_fov
# CAM_W_FOV = args.cam_w_fov
# TOF_H_FOV = args.tof_h_fov
# TOF_W_FOV = args.tof_w_fov
SET_TOF_MAX_DEPTH_TO_INVALID = args.set_tof_max_depth_to_invalid

# Options for sparse update
SU_UPDATE_ENC  = args.su_update_encoder
SU_UPDATE_DEC0 = args.su_update_dec0
SU_UPDATE_DEC1 = args.su_update_dec1
SU_UPDATE_DEC2 = args.su_update_dec2

# Statistics tracking
TRACK_EPOCHWISE_TEST_ACC = args.track_also_test_metrics


print("\n>>> INITIALIZING TRAINING <<<")

# Delete the tensorboard folder
if resume_from_checkpoint == False:
    if os.path.exists(tensorboard_folder):
        print("Removing old tensorboard folder before training..")
        shutil.rmtree(f"{tensorboard_folder}")

# Initialize tensorboard
writer = SummaryWriter(log_dir=tensorboard_folder)

# DATALOADERS
transform_train = True
transform_val   = False
transform_test  = False
normalize_imgs  = True
# Select if to hflip during training
HFLIP = True
if hflip_train == 0:
    HFLIP = False

idsiadepth_source_resolution = '48x48'

trainset    = idataloader.miniIDSIADepth(IDSIADEPTH_PATH, transform=transform_train, set='train', normalize=normalize_imgs, flip_horizontally=HFLIP, train_subset_percentage=PERCENTAGE_IDSIADEPTH_SAMPLES)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

if PERCENTAGE_IDSIADEPTH_SAMPLES < 100:
    print(f"Fine-tuning on a subset of {trainset.__len__()} samples ({PERCENTAGE_IDSIADEPTH_SAMPLES}% of the full IDSIA Depth)")

valset      = idataloader.miniIDSIADepth(IDSIADEPTH_PATH, transform=transform_val, set='val', normalize=normalize_imgs, flip_horizontally=False)
valloader   = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)

if TRACK_EPOCHWISE_TEST_ACC:
    print("Tracking also epochwise test accuracy..")
    testset      = idataloader.miniIDSIADepth(IDSIADEPTH_PATH, transform=transform_test, set='test', normalize=normalize_imgs, flip_horizontally=False)
    testloader   = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)


# Define device, models and training methods
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")


"""
GET SIZES OF INPUT DATA
"""

img_size = trainset.getimagesize()
dpt_size = trainset.getdepthsize()

IM_CH_IN = img_size[0]
IM_H_IN  = img_size[1]
IM_W_IN  = img_size[2]

DPTH_CH  = 1
DPTH_H   = dpt_size[0]
DPTH_W   = dpt_size[1]

print(f"\nTrain set size: {trainset.__len__()}")
print(f"Validation set size: {valset.__len__()}")

print(f"\nHorizontal flipping augmentation set to {HFLIP}.")


"""
MODEL DEFINITION AND INITIALIZATION
"""

if model_name == 'upydnet':

    net = uPydNet(IM_CH_IN, DPTH_CH).to(device)
    #for param in net.parameters():
    #    torch.nn.init.xavier_uniform_(param, gain=1.0, generator=None)

    # Load pre-trained model before all
    print("Loading pre-trained model..")
    net.load_state_dict(torch.load(f"{pre_trained_mdl_path}/{model_name}.pth"))

    # Print model structure
    print("\nModel to be trained:")
    summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=batch_size)
    print(f"\nInput size: [{IM_CH_IN}, {IM_H_IN}, {IM_W_IN}]")
    print(f"Output size: [{DPTH_CH}, {DPTH_H}, {DPTH_W}]")

    if DATA_TYPE == 'fp32':
        net = net.to(device)
    elif DATA_TYPE == 'fp16':
        net = net.half().to(device)
    elif DATA_TYPE == 'bfloat16':
        net = net.bfloat16().to(device)

    # Options for Sparse Update (only uPyD-Net)
    if SU_UPDATE_ENC == 0 or SU_UPDATE_DEC0 == 0 or SU_UPDATE_DEC1 == 0 or SU_UPDATE_DEC2 == 0:
        print("[Sparse Update] Training with Sparse Update")

        # Selectively freeze layers
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


elif model_name == 'upydnet_l':

    net = uPydNet_L(IM_CH_IN, DPTH_CH).to(device)

    #for param in net.parameters():
    #    torch.nn.init.xavier_uniform_(param, gain=1.0, generator=None)

    # Load pre-trained model before all
    print("Loading pre-trained model..")
    net.load_state_dict(torch.load(f"{pre_trained_mdl_path}/{model_name}.pth"))

    # Print model structure
    print("\nModel to be trained:")
    summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=batch_size)
    print(f"\nInput size: [{IM_CH_IN}, {IM_H_IN}, {IM_W_IN}]")
    print(f"Output size: [{DPTH_CH}, {DPTH_H}, {DPTH_W}]")

    if DATA_TYPE == 'fp32':
        net = net.to(device)
    elif DATA_TYPE == 'fp16':
        net = net.half().to(device)
    elif DATA_TYPE == 'bfloat16':
        net = net.bfloat16().to(device)

else:
    print('Invalid model selection!!')
    exit()



"""
TRAINING 
"""

learning_rate = startup_leaning_rate
if OPTIMIZER == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=10e-8)
elif OPTIMIZER == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=learning_rate) 

# Learning rate scheduling function
def change_lr_opt (optim, lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def get_lr_opt (optim):
    for param_group in optim.param_groups:
        print(param_group['lr'])


print(f"\nSetting optimizer to {OPTIMIZER}..")

schedule_threshold = 0

# Track best performance (initialize with bad values)
# VALUES: loss, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3
starting_epoch = 0
best_perf = [100] * 5
best_perf.append(0); best_perf.append(0); best_perf.append(0)
# If required, load checkpoint
if resume_from_checkpoint and os.path.exists(running_checkpoint_folder):
    print("Loading model from checkpoint...")
    net.load_state_dict(torch.load(f"{running_checkpoint_folder}/{model_name}.pth"))
    # Read performances from file. If best performance exists, load that, otherwise the running one
    if os.path.exists(checkpoint_folder):
        with open(f"{running_checkpoint_folder}/ckpt.info") as log:
            reader = csv.reader(log, delimiter=',')
            # epoch,learning_rate,test_loss,abs_rel,sq_rel,rmse,rmse_log,d1,d2,d3
            for row in reader:
                starting_epoch  = int(row[0])
                learning_rate   = float(row[1])
                best_perf[0]    = float(row[2])
                best_perf[1]    = float(row[3])
                best_perf[2]    = float(row[4])
                best_perf[3]    = float(row[5])
                best_perf[4]    = float(row[6])
                best_perf[5]    = float(row[7])
                best_perf[6]    = float(row[8])
                best_perf[7]    = float(row[9])
                break      
    elif os.path.exists(running_checkpoint_folder):  
        # Load the running one if the best doesn't exist
        with open(f"{running_checkpoint_folder}/ckpt.info") as log:
            reader = csv.reader(log, delimiter=',')
            # epoch,learning_rate,test_loss,abs_rel,sq_rel,rmse,rmse_log,d1,d2,d3
            for row in reader:
                starting_epoch  = int(row[0])
                learning_rate   = float(row[1])
                best_perf[0]    = float(row[2])
                best_perf[1]    = float(row[3])
                best_perf[2]    = float(row[4])
                best_perf[3]    = float(row[5])
                best_perf[4]    = float(row[6])
                best_perf[5]    = float(row[7])
                best_perf[6]    = float(row[8])
                best_perf[7]    = float(row[9])
                break
            

print("\nBeginning training procedure...")

if resume_from_checkpoint == False:
    with open(filename, 'w') as file:
        file.write("--------------------------------------------------------")
        file.write(f'\nTRAINING MODEL FOR {epochs} EPOCHS WITH LEARNING RATE {learning_rate}')    
        if PERCENTAGE_IDSIADEPTH_SAMPLES < 100:
            file.write(f'\nTRAINING ON A SUBSET OF {len(trainset)} SAMPLES ({PERCENTAGE_IDSIADEPTH_SAMPLES}%)')
        file.write("\n--------------------------------------------------------")

previous_val_loss = 100.0
best_epoch = 0

schedule_count = starting_epoch
for epoch in range(starting_epoch, epochs, 1):  # loop over the dataset multiple times

    # Manually schedule lr
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

        # Monitor learning rate
        writer.add_scalar("Learning_Rate/Epoch", learning_rate, epoch)

        # Get data from dictionary
        imgL    = data['img']
        disp    = data['disp']
        depth   = data['depth']
        fb      = data['fb']

        # import pdb; pdb.set_trace()

        if DATA_TYPE == 'fp32':
            imgL = imgL.type(torch.FloatTensor)
            imgL.to(device)
            disp.to(device)
            depth.to(device)
            fb.to(device)
        elif DATA_TYPE == 'fp16':
            imgL  = imgL.type(torch.HalfTensor)
            imgL  = imgL.to(device).type(torch.HalfTensor)
            disp  = disp.to(device).type(torch.HalfTensor)
            depth = depth.to(device).type(torch.HalfTensor)
            fb    = fb.to(device).type(torch.HalfTensor)
        elif DATA_TYPE == 'bfloat16':
            imgL  = imgL.type(torch.BFloat16Tensor)
            imgL  = imgL.to(device).bfloat16()
            disp  = disp.to(device).bfloat16()
            depth = depth.to(device).bfloat16()
            fb    = fb.to(device).bfloat16()

        if torch.sum(torch.isnan(imgL)) > 0:
            print('NaN values present in input!')


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(imgL.to(device)).to(device)
        outputs = torch.squeeze(outputs, 1)

        # import pdb; pdb.set_trace()

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

        """
        SET THE DESIRED SIZE FOR THE FINE-TUNING LABEL
        """

        # import pdb; pdb.set_trace()

        # figtest = plt.figure(figsize=(20,10), layout='constrained')
        # axstest = figtest.subplot_mosaic([['disp', 'rdisp(0)', 'rdisp(1)', 'rdisp(2)', 'rdisp(3)'],
        #                                   ['out',  'predd(0)', 'predd(1)', 'predd(2)', 'predd(3)']])
        # axstest['disp'].imshow(disp[0].cpu().float().detach().numpy())
        # axstest['out'].imshow(outputs[0].cpu().float().detach().numpy())

        if DOWNSAMPLE_PREDICTION_OR_UPSAMPLE_LABEL == 'UPSAMPLE_LABEL':
            with torch.no_grad():                
                # Upsample to provide the upscaled proxy label
                depth = upsample_tof_data(depth, size=[48, 48], mode=UPSAMPLE_STRATEGY)
                disp  = upsample_tof_data(disp, size=[48, 48], mode=UPSAMPLE_STRATEGY)
        if DOWNSAMPLE_PREDICTION_OR_UPSAMPLE_LABEL == 'DOWNSAMPLE_PREDICTION':
                print("[DOWNSAMPLE_PREDICTION] ROUTINE NOT CHECKED!!")
                pool_ker_size = int(48 / 8)
                pool_ker_stride = int(48 / 8)
                downsampler   = nn.MaxPool2d(pool_ker_size, pool_ker_stride).to(device)                    
                disp = simulate_tof_sensor_disparity(disp, 8, 8, 8, 8, -1, 'nearest')
                red_outputs = downsampler(outputs)

        if SET_TOF_MAX_DEPTH_TO_INVALID == 1:
            mask = (depth == 4.0)
            depth[mask] = -1
            disp[mask] = -1
            #import pdb; pdb.set_trace()

        # axstest['rdisp(0)'].imshow(disp[0][0].cpu().float().detach().numpy()); axstest['rdisp(1)'].imshow(disp[0][1].cpu().float().detach().numpy()) 
        # axstest['rdisp(2)'].imshow(disp[0][2].cpu().float().detach().numpy()); axstest['rdisp(3)'].imshow(disp[0][3].cpu().float().detach().numpy())
        # axstest['predd(0)'].imshow(red_outputs[0][0].cpu().float().detach().numpy()); axstest['predd(1)'].imshow(red_outputs[0][1].cpu().float().detach().numpy())
        # axstest['predd(2)'].imshow(red_outputs[0][2].cpu().float().detach().numpy()); axstest['predd(3)'].imshow(red_outputs[0][3].cpu().float().detach().numpy())

        # plt.show()

        # import pdb; pdb.set_trace()


                
        # with torch.no_grad():
        #     print("Training!")
        #     for i in range(batch_size):
        #         fig, ax = plt.subplots(1, 3)

        #         imgsh  = imgL[i].squeeze(0).permute(1,2,0).to('cpu').type(torch.uint8)
        #         dispsh = disp[i].squeeze(0).to('cpu')
        #         outsh  = outputs[i].squeeze(0).to('cpu')
        #         ax[0].imshow(imgsh)
        #         ax[1].imshow(dispsh)
        #         ax[2].imshow(outsh)

        #         plt.tight_layout()
        #         plt.show()

        # loss, Lap, Lps, La, Lb = MonocularDepthSemiSupervisedLoss(
        #             output         = outputs.to(device), 
        #             target         = disp.to(device), 
        #             imgL           = imgL.to(device), 
        #             imgR           = imgR.to(device), 
        #             aap            = aap, 
        #             aps            = aps,
        #             alpha          = 0.2,
        #             invalid_value  = -1,
        #             original_width = MINIKITTI_MAX_WIDTH,
        #             device         = device,
        #             data_type      = DATA_TYPE)
        
        if DOWNSAMPLE_PREDICTION_OR_UPSAMPLE_LABEL == 'DOWNSAMPLE_PREDICTION':
            loss = ProxySupervisionLoss(
                output        = red_outputs.to(device),
                target        = disp.to(device),
                alpha         = 0.2,
                invalid_value = -1,
                device        = device,
                data_type     = DATA_TYPE
            )
        else:
            loss = ProxySupervisionLoss(
                output        = outputs.to(device),
                target        = disp.to(device),
                alpha         = 0.2,
                invalid_value = -1,
                device        = device,
                data_type     = DATA_TYPE
            )
        Lap = 0; Lps = 0; La = 0; Lb = 0

        loss.backward()
        optimizer.step()

        # import pdb; pdb.set_trace()

        with torch.no_grad():
            # print statistics
            # Indices
            if PERCENTAGE_IDSIADEPTH_SAMPLES < 100:
                check_idx = 10 #int(len(trainset) / batch_size)
            else:
                check_idx = 50 #((len(trainset)+1) / batch_size) #/ 100
            if i % check_idx == check_idx-1:    
                print(f"[Epoch:{epoch}, batch:{i + 1:5d}] train_loss = {loss:.3f}") #, where Lap = {Lap:.3f} (Lssim = {La:.3f}, Lmod = {Lb:.3f}), Lps = {Lps:.3f}")   

    if SCHEDULE_LR:
        schedule_count += 1


    """
    VALIDATE AFTER EVERY EPOCH
    """

    total = 0
    val_loss = 0; val_Lap = 0; val_Lps = 0; val_La = 0; val_Lb = 0
    num_batches = len(valloader)
    size = len(valloader.dataset)
    abs_rel=0; sq_rel=0; rmse=0; rmse_log=0; d1=0; d2=0; d3=0
    silog = 0

    last_val_output = 0
    
    net.eval()
    with torch.no_grad():
        for test_data in valloader:

            # Get data from dictionary
            val_imgL    = test_data['img']
            val_disp    = test_data['disp']
            val_depth   = test_data['depth']
            val_fb      = test_data['fb']

            if DATA_TYPE == 'fp32':
                val_imgL    = val_imgL.type(torch.FloatTensor)
                val_imgL    = val_imgL.to(device)
                val_disp    = val_disp.to(device)
                val_depth   = val_depth.to(device)
                val_fb      = val_fb.to(device)
            elif DATA_TYPE == 'fp16':
                val_imgL    = val_imgL.type(torch.HalfTensor)
                val_imgL    = val_imgL.to(device).half()
                val_disp    = val_disp.to(device).half()
                val_depth   = val_depth.to(device).half()
                val_fb      = val_fb.to(device).half()
            elif DATA_TYPE == 'bfloat16':
                val_imgL    = val_imgL.type(torch.BFloat16Tensor)
                val_imgL    = val_imgL.to(device).bfloat16()
                val_disp    = val_disp.to(device).bfloat16()
                val_depth   = val_depth.to(device).bfloat16()
                val_fb      = val_fb.to(device).bfloat16()

            # calculate outputs by running images through the network
            val_outputs = net(val_imgL.to(device)).to(device)
            val_outputs = torch.squeeze(val_outputs, 1)

            if ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 1:
                val_depth = val_depth[:, 1:7, 0:7]
                val_disp  = val_disp[:, 1:7, 0:7]

                # figx, axx = plt.subplots(3)
                # axx[0].imshow(imgL[0].float().cpu().permute(1,2,0).numpy())
                # axx[1].imshow(depth[0].float().cpu().numpy())
                # axx[2].imshow(disp[0].float().cpu().numpy())
                # plt.show()
                # import pdb; pdb.set_trace()
            elif ALIGN_CAM_TOF_FOV == 1 and CROP_BORDER_TOF_VALUES == 0:
                pass

            # Upsample ToF labels with nearest neighbour to evaluate the model
            val_disp = upsample_tof_data(val_disp, size=[48, 48], mode='nearest')

            # val_loss_t, val_Lap_t, val_Lps_t, val_La_t, val_Lb_t = MonocularDepthSemiSupervisedLoss(
            #                                         output         = val_outputs.to(device), 
            #                                         target         = val_disp.to(device), 
            #                                         imgL           = val_imgL.to(device), 
            #                                         imgR           = val_imgR.to(device), 
            #                                         aap            = aap, 
            #                                         aps            = aps,
            #                                         alpha          = 0.2,
            #                                         invalid_value  = -1,
            #                                         original_width = MINIKITTI_MAX_WIDTH,
            #                                         device         = device,
            #                                         data_type      = DATA_TYPE)

            val_loss_t = ProxySupervisionLoss(
                output        = val_outputs.to(device),
                target        = val_disp.to(device),
                alpha         = 0.2,
                invalid_value = -1,
                device        = device,
                data_type     = DATA_TYPE
            )

            val_loss += val_loss_t
            # val_Lap  += 0 # val_Lap_t
            # val_Lps  += 0 # val_Lps_t
            # val_La   += 0 # val_La_t
            # val_Lb   += 0 # val_Lb_t

            # Compute metrics
            val_out_depth_ups = upsample_tof_data(val_depth, size=[48, 48], mode='nearest')
            val_outputs_depth = compute_depth_map_test(val_fb, val_outputs, device, DATA_TYPE)
            run_abs_rel, run_sq_rel, run_rmse, run_rmse_log, run_d1, run_d2, run_d3 = compute_errors_masked_train(val_out_depth_ups.to(device), val_outputs_depth)
            run_silog = ScaleInvariantMSELogLoss(val_out_depth_ups.to(device), val_outputs_depth, 1.0)
            abs_rel  += run_abs_rel
            sq_rel   += run_sq_rel
            rmse     += run_rmse
            rmse_log += run_rmse_log
            d1       += run_d1
            d2       += run_d2
            d3       += run_d3
            silog    += run_silog

            # import pdb; pdb.set_trace()

            # with torch.no_grad():
            #     print("Validation!")
            #     for i in range(batch_size):
            #         fig, ax = plt.subplots(1, 4)

            #         img  = val_imgL[i].squeeze(0).permute(1,2,0).to('cpu').type(torch.uint8)
            #         disp = val_disp[i].squeeze(0).to('cpu')
            #         out  = val_out_disp_ups[i].squeeze(0).to('cpu')
            #         out_depth = val_out_depth_ups[i].squeeze(0).to('cpu')
            #         ax[0].imshow(img)
            #         ax[1].imshow(disp)
            #         ax[2].imshow(out)
            #         ax[3].imshow(out_depth)

            #         plt.tight_layout()
            #         plt.show()

            #         import pdb; pdb.set_trace()

            # Save validation output to be printed
            last_val_output = val_outputs

            # import pdb; pdb.set_trace()

    with torch.no_grad():
        val_loss    /= num_batches
        abs_rel     /= num_batches
        sq_rel      /= num_batches
        rmse        /= num_batches
        rmse_log    /= num_batches
        d1          /= num_batches
        d2          /= num_batches
        d3          /= num_batches
        silog       /= num_batches

        with open(filename, 'a') as file:
            file.write(f"\n>>> EPOCH {epoch} <<<\n")
            file.write(f"LR = {learning_rate}, avg_loss = {val_loss:>8f}\n")
            file.write(f"abs_rel   = {abs_rel:.3f}\n")
            file.write(f"sq_rel    = {sq_rel:.3f}\n")
            file.write(f"rmse      = {rmse:.3f}\n")
            file.write(f"rmse_log  = {rmse_log:.3f}\n")
            file.write(f"d1        = {d1:.3f}\n")
            file.write(f"d2        = {d2:.3f}\n")
            file.write(f"d3        = {d3:.3f}\n")
            file.write(f"silogloss = {silog:.3f}\n")

        with open(out_file, 'w') as file:
            file.write(f"\n>>> Epoch {epoch}, Validation output:\n")
            file.write(f"{dmp.tensor_to_string(last_val_output)}")
        
        """
        TRACK LOSS AND QUALITY METRICS AFTER EACH EPOCH
        """

        # Loss components (train)
        writer.add_scalar("Train_Loss/Epoch", loss, epoch)
        # writer.add_scalar("Train_Lap/Epoch" , Lap, epoch)
        # writer.add_scalar("Train_Lps/Epoch" , Lps, epoch)
        # Loss components (validation)
        writer.add_scalar("Val_Loss/Epoch", val_loss, epoch)
        # writer.add_scalar("Val_Lap/Epoch" , val_Lap, epoch)
        # writer.add_scalar("Val_Lps/Epoch" , val_Lps, epoch)
        # Qualiy metrics (validation)
        writer.add_scalar("Abs_rel/Epoch" , abs_rel, epoch)
        writer.add_scalar("Sq_rel/Epoch"  , sq_rel, epoch)
        writer.add_scalar("RMSE/Epoch"    , rmse, epoch)
        writer.add_scalar("RMSE_log/Epoch", rmse_log, epoch)
        writer.add_scalar("d1/Epoch"      , d1, epoch)
        writer.add_scalar("d2/Epoch"      , d2, epoch)
        writer.add_scalar("d3/Epoch"      , d3, epoch)
        writer.add_scalar("silogloss/Epoch", silog, epoch)
        if TRACK_EPOCHWISE_TEST_ACC == 0:
            # Flush events on disk
            writer.flush()

        # import pdb; pdb.set_trace()

        # # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
        # if EARLY_STOPPING == 1:
        #     if epoch == starting_epoch:
        #         previous_val_loss = val_loss
        #     # Find improvement with respect to previous loss
        #     elif epoch > 0:
        #         # Check if the metrics have been enhanced and add them to the pool
        #         if ((abs_rel < best_perf[1] and sq_rel < best_perf[2] and rmse < best_perf[3] and rmse_log < best_perf[4]) 
        #             or (d1 > best_perf[5] and d2 > best_perf[6] and d3 > best_perf[7])):
        #             statistics_enhanced = True
        #         else: 
        #             statistics_enhanced = False
        #         # Check loss
        #         loss_difference = (previous_val_loss - val_loss)
        #         print(f"Early stopping: {loss_difference:.5f} between current ({val_loss:.3f}) and previous ({previous_val_loss:.3f}) losses, statistics enhanced? {statistics_enhanced}.")
        #         if val_loss < previous_val_loss or statistics_enhanced == True:
        #             previous_val_loss = val_loss
        #             patience_counter  = 0
        #         elif val_loss > (previous_val_loss + PATIENCE_TOLERANCE) or statistics_enhanced == False: 
        #             patience_counter += 1
        #             print(f"Validation loss increase detected (patience counter {patience_counter}/{PATIENCE})!")
        #     if patience_counter == PATIENCE:
        #         STOP_TRAINING = True

        """
        IF SELECTED, PERFORM A TEST IN EACH EPOCH TO TRACK EVOLUTION (ON-DEVICE LEARNING EXPERIMENT)
        """
        if TRACK_EPOCHWISE_TEST_ACC == 1:

            print(f"Testing epoch {epoch}..")

            total_test = 0
            num_test_batches = len(testloader)
            test_size = len(testloader.dataset)

            # Track the statistics with respect to the whole test set
            test_loss_list = []; #test_Lap_list = []; test_Lps_list = []; test_La_list = []; test_Lb_list = []
            test_abs_rel_list=[]; test_sq_rel_list=[]; test_rmse_list=[]; test_rmse_log_list=[]; test_d1_list=[]; test_d2_list=[]; test_d3_list=[]; test_silog_list=[]

            # Statistics vs KITTI ground truth
            test_loss = 0; test_Lap = 0; test_Lps = 0; test_La = 0; test_Lb = 0
            test_abs_rel=0; test_sq_rel=0; test_rmse=0; test_rmse_log=0; test_d1=0; test_d2=0; test_d3=0; test_silog=0

            with torch.no_grad():
                for test_data in testloader:

                    # Get data from dictionary
                    test_imgL    = test_data['img']
                    test_disp    = test_data['disp']
                    test_depth   = test_data['depth']
                    test_fb      = test_data['fb']

                    if DATA_TYPE == 'fp32':
                        test_imgL = test_imgL.type(torch.FloatTensor)
                        test_imgL.to(device)
                        test_disp.to(device)
                        test_depth.to(device)
                        test_fb.to(device)
                    elif DATA_TYPE == 'fp16':
                        test_imgL = test_imgL.type(torch.HalfTensor)
                        test_imgL.to(device).half()
                        test_disp.to(device).half()
                        test_depth.to(device).half()
                        test_fb.to(device).half()
                    elif DATA_TYPE == 'bfloat16':
                        test_imgL = test_imgL.type(torch.BFloat16Tensor)
                        test_imgL.to(device).bfloat16()
                        test_disp.to(device).bfloat16()
                        test_depth.to(device).bfloat16()
                        test_fb.to(device).bfloat16()

                    # calculate outputs by running images through the network
                    test_outputs = net(test_imgL.to(device)).to(device)

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

                    # Add post-processing from https://openaccess.thecvf.com/content_cvpr_2017/papers/Godard_Unsupervised_Monocular_Depth_CVPR_2017_paper.pdf
                    flipped_imgL      = tfun.hflip(test_imgL)
                    test_outputs_flipped_p = net(flipped_imgL.to(device)).to(device)
                    test_outputs_flipped   = tfun.hflip(test_outputs_flipped_p)

                    # Average outputs and assign borders
                    border_size  = int(test_imgL.size()[-1] * 0.05)
                    img_width    = test_imgL.size()[-1]
                    test_outputs_filt = (test_outputs + test_outputs_flipped) / 2
                    test_outputs_filt[:, :, 0:border_size]    = test_outputs_flipped[:, :, 0:border_size]
                    test_outputs_filt[:, :, (img_width-1):-1] = test_outputs[:, :, (img_width-1):-1]

                    # Squeeze results
                    test_outputs_filt = torch.squeeze(test_outputs_filt, 1)

                    # Compute depth map to compare with ToF data and upscale ToF data to align to prediction
                    test_outputs_depth = compute_depth_map_test(test_fb, test_outputs_filt, device, 'fp32').type(torch.FloatTensor)
                    test_depth = upsample_tof_data(test_depth, size=[48, 48], mode='nearest').type(torch.FloatTensor)

                    IDSIADEPTH_MAX_WIDTH = 240

                    if SET_TOF_MAX_DEPTH_TO_INVALID == 1:
                        vmask = (test_depth == 4.0)
                        test_depth[vmask] = -1
                        #test_disp[mask] = -1
                        #import pdb; pdb.set_trace()

                    """
                    STATISTICS WITH RESPECT TO GROUND TRUTH
                    """

                    test_loss_t = Test_ProxySupervisionLoss(
                        output        = test_outputs_depth.to(device),
                        target        = test_depth.to(device),
                        alpha         = 0.2,
                        invalid_value = -1,
                        device        = device,
                        data_type     = DATA_TYPE
                    )
                    test_loss += test_loss_t;   test_loss_list.append(float(test_loss_t.to('cpu')))
                    # test_Lap  += 0
                    # test_Lps  += 0
                    # test_La   += 0
                    # test_Lb   += 0

                    run_abs_rel, run_sq_rel, run_rmse, run_rmse_log, run_d1, run_d2, run_d3 = compute_errors_masked_train(test_depth.to(device), test_outputs_depth.to(device))
                    run_silog = ScaleInvariantMSELogLoss(test_depth.to(device), test_outputs_depth.to(device), 1.0)
                    test_abs_rel  += run_abs_rel     ; test_abs_rel_list.append(float(run_abs_rel.to('cpu')))
                    test_sq_rel   += run_sq_rel      ; test_sq_rel_list.append(float(run_sq_rel.to('cpu')))
                    test_rmse     += run_rmse        ; test_rmse_list.append(float(run_rmse.to('cpu')))
                    test_rmse_log += run_rmse_log    ; test_rmse_log_list.append(float(run_rmse_log.to('cpu')))
                    test_d1       += run_d1          ; test_d1_list.append(float(run_d1.to('cpu')))
                    test_d2       += run_d2          ; test_d2_list.append(float(run_d2.to('cpu')))
                    test_d3       += run_d3          ; test_d3_list.append(float(run_d3.to('cpu')))
                    test_silog    += run_silog       ; test_silog_list.append(float(run_silog.to('cpu')))


                """
                STATISTICS WITH RESPECT TO GROUND TRUTH
                """

                test_loss        /= num_test_batches
                test_abs_rel     /= num_test_batches
                test_sq_rel      /= num_test_batches
                test_rmse        /= num_test_batches
                test_rmse_log    /= num_test_batches
                test_d1          /= num_test_batches
                test_d2          /= num_test_batches
                test_d3          /= num_test_batches
                test_silog       /= num_test_batches

                # Loss components (test)
                writer.add_scalar("Test_Loss/Epoch", test_loss, epoch)
                # Qualiy metrics (validation)
                writer.add_scalar("Test_Abs_rel/Epoch"  , test_abs_rel, epoch)
                writer.add_scalar("Test_Sq_rel/Epoch"   , test_sq_rel, epoch)
                writer.add_scalar("Test_RMSE/Epoch"     , test_rmse, epoch)
                writer.add_scalar("Test_RMSE_log/Epoch" , test_rmse_log, epoch)
                writer.add_scalar("Test_d1/Epoch"       , test_d1, epoch)
                writer.add_scalar("Test_d2/Epoch"       , test_d2, epoch)
                writer.add_scalar("Test_d3/Epoch"       , test_d3, epoch)
                writer.add_scalar("Test_silogloss/Epoch", test_silog, epoch)
                # Flush events on disk
                writer.flush()

                print(f"Testing complete!")

        """
        CHECKPOINT MODEL AND CHECK IF THE BEST ACCURACY IS REACHED
        """

        checkpoint_model = False
        # best_epoch = 0
        # # Check if best accuracy is reached
        # if ((val_loss > 0 and abs_rel >= 0 and sq_rel >= 0 and rmse >= 0 and rmse_log >= 0 and d1 >= 0 and d2 >= 0 and d3 >= 0) and
        #         (abs_rel < best_perf[1] and sq_rel < best_perf[2] and rmse < best_perf[3] 
        #         and rmse_log < best_perf[4]) or (d1 > best_perf[5] and d2 > best_perf[6] 
        #         and d3 > best_perf[7])):
        if (val_loss >= 0) and (val_loss < previous_val_loss):
            checkpoint_model = True
            best_perf = [val_loss, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3]
            best_epoch = epoch
            previous_val_loss = val_loss

        # Save best performing model during training
        if checkpoint_model:
            print(f">>> Checkpointing best performing model to {checkpoint_folder} at epoch {epoch}...")
            if not os.path.exists(checkpoint_folder):
                os.mkdir(checkpoint_folder)
            # Model info
            with open(f"{checkpoint_folder}/model.info", 'w') as log:
                log.write(f"--- ckpt.info organization (tracks best epoch): ---\n")
                log.write(f"epoch,learning_rate,")
                log.write(f"test_loss,abs_rel,sq_rel,rmse,rmse_log,d1,d2,d3\n")
                #original_stdout = sys.stdout
                #sys.stdout = log
                #summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=batch_size)
                #sys.stdout = original_stdout
            # Checkpoint epoch and metrics
            with open(f"{checkpoint_folder}/ckpt.info", 'w') as log:
                log.write(f"{best_epoch},{learning_rate},")
                log.write(f"{best_perf[0]},{best_perf[1]},{best_perf[2]},{best_perf[3]},{best_perf[4]},{best_perf[5]},{best_perf[6]},{best_perf[7]}\n")
            # Save optimal model
            torch.save(net.state_dict(), f"{checkpoint_folder}/{model_name}.pth")

        # Save also the model every epoch, regardless of its best performance
        print(f">>> Saving current epoch's model to {running_checkpoint_folder} at epoch {epoch}...")
        if not os.path.exists(running_checkpoint_folder):
            os.mkdir(running_checkpoint_folder)
        # Model info
        with open(f"{running_checkpoint_folder}/model.info", 'w') as log:
            log.write(f"--- ckpt.info organization (tracks last training epoch): ---\n")
            log.write(f"epoch,learning_rate,")
            log.write(f"test_loss,abs_rel,sq_rel,rmse,rmse_log,d1,d2,d3\n")
            #original_stdout = sys.stdout
            #sys.stdout = log
            #summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=batch_size)
            #sys.stdout = original_stdout
        # Checkpoint epoch and metrics
        with open(f"{running_checkpoint_folder}/ckpt.info", 'w') as log:
            log.write(f"{epoch},{learning_rate},")
            log.write(f"{val_loss},{abs_rel},{sq_rel},{rmse},{rmse_log},{d1},{d2},{d3}\n")
        # Save optimal model
        torch.save(net.state_dict(), f"{running_checkpoint_folder}/{model_name}.pth")

net.eval()

with torch.no_grad():
    """
    SAVE TRAINED MODEL (STORE THE BEST PERFORMING)
    """

    print("Finished training of the full model!")

    if SAVE_MODEL:
        print(f"Saving full model's weights after training to {statedict_file}..")
        # Save optimal model
        if os.path.exists(checkpoint_folder):
            shutil.copyfile(f"{checkpoint_folder}/{model_name}.pth", statedict_file)
            copy_tree(f"{tensorboard_folder}/", f"{statedict_dir}/tensorboard_data/")
        else:
            # Save last epoch's model
            torch.save(net.state_dict(), statedict_file)
            copy_tree(f"{tensorboard_folder}/", f"{statedict_dir}/tensorboard_data/")
        with open(statedict_info, 'w') as f:
            f.write("--- Input and label sizes: ---\n")
            f.write(f"Input size: [{IM_CH_IN}, {IM_H_IN}, {IM_W_IN}]\n")
            f.write(f"Label size: [{DPTH_CH}, {DPTH_H}, {DPTH_W}]\n")
            f.write("\n--- Hyperparameters: ---\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Learning rate: {initial_learning_rate} to {learning_rate} (scale every {scheduler_epochs_step} epochs)\n")
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
            #original_stdout = sys.stdout
            #sys.stdout = f
            #summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=batch_size)
            #print("\n\n")
            #printed_model = net.to('cpu')
            #stat(printed_model, (IM_CH_IN, IM_H_IN, IM_W_IN))
            #sys.stdout = original_stdout

    """
    DELETE CHECKPOINT FILE
    """

    if DELETE_CKPT_AFTER_TRAINING:
        print("Removing checkpoint folder after training..")
        os.remove(f"{checkpoint_folder}")
            
    # Close tensorboard
    writer.close()
