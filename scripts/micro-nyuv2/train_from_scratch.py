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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import sys
import os
import shutil
import csv
from distutils.dir_util import copy_tree
from torchsummary import summary
from torchstat import stat
from utils.evaluation import *
# Network definitions
from models.upydnet import *
import utils.dataloader as dataloader
# Custom loss
from utils.losses import *
from utils.processing import *
import utils.dump_utils as dmp
# Data visualization
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Parser
parser = argparse.ArgumentParser("Monocular Depth Estimation CNN Training - uPyD-Net")
# Dataset and model setup
parser.add_argument( '--mininyuv2_path', type=str, default='../../micro-nyuv2/')
parser.add_argument( '--mini_nyuv2_highest_width_resolution', type=int, default=360)  # Needed to compute the rescaling of disparity maps in image reprojection loss
parser.add_argument( '--model_name', type=str, default='upydnet')                     # 'cnn' or 'upydnet' or 'upydnet_l'
# Training setup
parser.add_argument( '--epochs', type=int, default=100 )
parser.add_argument( '--batch_size', type=int, default=16 )
parser.add_argument( '--startup_learning_rate', type=float, default=1e-4)    # First epoch's learning rate to avoid gradient explosion
parser.add_argument( '--startup_epoch', type=int, default=40)                 # Epoch after which to apply the init_learning_rate
parser.add_argument( '--init_learning_rate', type=float, default=0.5e-4)        # Initial learning rate after first startup epochs
parser.add_argument( '--scheduler_epochs_step', type=int, default=40)
parser.add_argument( '--schedule_lr', type=bool, default=False)
parser.add_argument( '--flip_horizontally_train', type=int, default=1)          # Select if to flip or not the images and labels during training
# Training with partial data
parser.add_argument( '--percentage_nyuv2_trainset', type=int, default=100)             # Specify the percentage of training data in the KITTI train set to be used in training
# Loss tuning parameters
# parser.add_argument( '--loss_aap', type=float, default=0.0)     # Tuning parameter for the self-supervised part of the loss
# parser.add_argument( '--loss_aps', type=float, default=1.0)     # Tuning parameter for the proxy-supervised part of the loss
# parser.add_argument( '--epoch_set_loss_params_to_init', type=int, default=0)    # Epoch at which to set the loss tuning parameters to the selected ones
# Saving and resume setup
parser.add_argument( '--save_trained_mdl', type=bool, default=True)
parser.add_argument( '--saved_mdl_path', type=str, default='checkpoints/')
parser.add_argument( '--resume_from_checkpoint', type=bool, default=False)
# Other folders (checkpoint, tensorboard)
parser.add_argument( '--checkpoint_folder', type=str, default='./ckpt')                 # Folder for the best performing model
parser.add_argument( '--running_checkpoint_folder', type=str, default='./ckpt_run')     # Folder with last epoch's model
parser.add_argument( '--tensorboard_folder', type=str, default='./run')                 # Folder to store tensorboard data about training
args = parser.parse_args()

# Training hyperparameters and misc
MININYUV2_PATH = args.mininyuv2_path
model_name = args.model_name
epochs = args.epochs
batch_size = args.batch_size
# As 48 disparity maps are produced downsampling higher resolution disparity maps (e.g., 360 in width)
# this is needed to correctly scale the warping in the PhotometricReprojectionLoss 
MININYUV2_MAX_WIDTH = args.mini_nyuv2_highest_width_resolution 
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
# loss_aap = args.loss_aap
# loss_aps = args.loss_aps
# epoch_set_loss_params_to_init = args.epoch_set_loss_params_to_init
SCHEDULE_LR = args.schedule_lr
SAVE_MODEL = args.save_trained_mdl
# Set to True if you want to delete ckpt after training
DELETE_CKPT_AFTER_TRAINING = False

# Option for partial data training
PERCENTAGE_NYUV2_SAMPLES = args.percentage_nyuv2_trainset


print("\n>>> INITIALIZING TRAINING <<<")

# Delete the tensorboard folder
if resume_from_checkpoint == False and os.path.exists(f'{tensorboard_folder}'):
    if os.path.exists('.run/'):
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

nyuv2_source_resolution = '48x48'

trainset    = dataloader.miniNYUv2(MININYUV2_PATH, transform=transform_train, set='train', normalize=normalize_imgs, flip_horizontally=HFLIP, train_subset_percentage=PERCENTAGE_NYUV2_SAMPLES)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

if PERCENTAGE_NYUV2_SAMPLES < 100:
    print(f"Fine-tuning on a subset of {trainset.__len__()} samples ({PERCENTAGE_NYUV2_SAMPLES}% of the full mini-nyuv2)")

valset      = dataloader.miniNYUv2(MININYUV2_PATH, transform=transform_val, set='val', normalize=normalize_imgs, flip_horizontally=False)
valloader   = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)


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

    # Print model structure
    print("\nModel to be trained:")
    summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=batch_size)
    print(f"\nInput size: [{IM_CH_IN}, {IM_H_IN}, {IM_W_IN}]")
    print(f"Output size: [{DPTH_CH}, {DPTH_H}, {DPTH_W}]")

elif model_name == 'upydnet_l':

    net = uPydNet_L(IM_CH_IN, DPTH_CH).to(device)
    #for param in net.parameters():
    #    torch.nn.init.xavier_uniform_(param, gain=1.0, generator=None)

    # Print model structure
    print("\nModel to be trained:")
    summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=batch_size)
    print(f"\nInput size: [{IM_CH_IN}, {IM_H_IN}, {IM_W_IN}]")
    print(f"Output size: [{DPTH_CH}, {DPTH_H}, {DPTH_W}]")

else:
    print('Invalid model selection!!')
    exit()



"""
TRAINING 
"""

learning_rate = startup_leaning_rate
optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=10e-8)

# Learning rate scheduling function
def change_lr_opt (optim, lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def get_lr_opt (optim):
    for param_group in optim.param_groups:
        print(param_group['lr'])

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

with open(filename, 'w') as file:
    file.write("--------------------------------------------------------")
    file.write(f'\nTRAINING MODEL FOR {epochs} EPOCHS WITH LEARNING RATE {learning_rate}')    
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

    # # Schedule loss parameters according to epochs
    # if epoch < epoch_set_loss_params_to_init:
    #     aap = 0.5
    #     aps = 0.5
    #     print(f"Setting loss parameters to aap = {aap}, aps = {aps}")
    # elif epoch >= epoch_set_loss_params_to_init:
    #     aap = loss_aap
    #     aps = loss_aps
    #     print(f"Setting loss parameters to aap = {aap}, aps = {aps}")        

    """
    TRAIN MODEL FOR CURRENT EPOCH
    """

    running_loss = 0.0
    net.train()

    for i, data in enumerate(trainloader):

        # Monitor learning rate
        writer.add_scalar("Learning_Rate/Epoch", learning_rate, epoch)

        # Get data from dictionary
        imgL    = data['imgL']
        disp    = data['dispL']
        depth   = data['depthL']
        fb      = data['fb']

        imgL = imgL.type(torch.FloatTensor)
        imgL.to(device)
        disp.to(device)
        depth.to(device)
        fb.to(device)

        if torch.sum(torch.isnan(imgL)) > 0:
            print('NaN values present in input!')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(imgL.to(device)).to(device)
        outputs = torch.squeeze(outputs, 1)

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
        #             imgR           = None, #imgR.to(device), 
        #             aap            = aap, 
        #             aps            = aps,
        #             alpha          = 0.2,
        #             invalid_value  = -1,
        #             original_width = MININYUV2_MAX_WIDTH,
        #             device         = device)

        loss = ProxySupervisionLoss(
            output        = outputs.to(device),
            target        = disp.to(device),
            alpha         = 0.2,
            invalid_value = -1,
            device        = device
        )
        Lap = 0
        Lps = loss
        La  = 0; Lb = 0

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # print statistics
            # Indices
            check_idx = 40 #((len(trainset)+1) / batch_size)
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

    last_val_output = 0
    
    net.eval()
    with torch.no_grad():
        for test_data in valloader:

            # Get data from dictionary
            val_imgL    = test_data['imgL']
            val_depthGT = test_data['depthGT']
            val_disp    = test_data['dispL']
            val_depth   = test_data['depthL']
            val_fb      = test_data['fb']

            val_imgL = val_imgL.type(torch.FloatTensor)
            val_imgL.to(device)
            val_depthGT.to(device)
            val_disp.to(device)
            val_depth.to(device)
            val_fb.to(device)

            # calculate outputs by running images through the network
            val_outputs = net(val_imgL.to(device)).to(device)
            val_outputs = torch.squeeze(val_outputs, 1)

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
            #                                         device         = device)
            # val_loss += val_loss_t
            # val_Lap  += val_Lap_t
            # val_Lps  += val_Lps_t
            # val_La   += val_La_t
            # val_Lb   += val_Lb_t

            val_loss_t = ProxySupervisionLoss(
                output        = val_outputs.to(device),
                target        = val_disp.to(device),
                alpha         = 0.2,
                invalid_value = -1,
                device        = device
            )
            val_loss += val_loss_t
            val_Lap  += 0
            val_Lps  += 0
            val_La   += 0
            val_Lb   += 0

            # Compute metrics
            val_out_disp_ups  = transforms.functional.resize(img=val_outputs, size=[val_depthGT.size()[-3], val_depthGT.size()[-2]], interpolation=transforms.InterpolationMode.NEAREST)
            val_out_depth_ups = compute_depth_map_validation(val_fb, val_out_disp_ups, device)
            run_abs_rel, run_sq_rel, run_rmse, run_rmse_log, run_d1, run_d2, run_d3 = compute_errors_masked_train(val_out_depth_ups.to(device), val_depthGT.to(device).squeeze(-1))
            abs_rel  += run_abs_rel
            sq_rel   += run_sq_rel
            rmse     += run_rmse
            rmse_log += run_rmse_log
            d1       += run_d1
            d2       += run_d2
            d3       += run_d3

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

    with torch.no_grad():
        val_loss    /= num_batches
        abs_rel     /= num_batches
        sq_rel      /= num_batches
        rmse        /= num_batches
        rmse_log    /= num_batches
        d1          /= num_batches
        d2          /= num_batches
        d3          /= num_batches

        with open(filename, 'a') as file:
            file.write(f"\n>>> EPOCH {epoch} <<<\n")
            file.write(f"LR = {learning_rate}, avg_loss = {val_loss:>8f}\n")
            file.write(f"abs_rel  = {abs_rel:.3f}\n")
            file.write(f"sq_rel   = {sq_rel:.3f}\n")
            file.write(f"rmse     = {rmse:.3f}\n")
            file.write(f"rmse_log = {rmse_log:.3f}\n")
            file.write(f"d1       = {d1:.3f}\n")
            file.write(f"d2       = {d2:.3f}\n")
            file.write(f"d3       = {d3:.3f}\n")

        with open(out_file, 'w') as file:
            file.write(f"\n>>> Epoch {epoch}, Validation output:\n")
            file.write(f"{dmp.tensor_to_string(last_val_output)}")
        
        """
        TRACK LOSS AND QUALITY METRICS AFTER EACH EPOCH
        """

        # Loss components (train)
        writer.add_scalar("Train_Loss/Epoch", loss, epoch)
        writer.add_scalar("Train_Lap/Epoch" , Lap, epoch)
        writer.add_scalar("Train_Lps/Epoch" , Lps, epoch)
        # Loss components (validation)
        writer.add_scalar("Val_Loss/Epoch", val_loss, epoch)
        writer.add_scalar("Val_Lap/Epoch" , val_Lap, epoch)
        writer.add_scalar("Val_Lps/Epoch" , val_Lps, epoch)
        # Qualiy metrics (validation)
        writer.add_scalar("Abs_rel/Epoch" , abs_rel, epoch)
        writer.add_scalar("Sq_rel/Epoch"  , sq_rel, epoch)
        writer.add_scalar("RMSE/Epoch"    , rmse, epoch)
        writer.add_scalar("RMSE_log/Epoch", rmse_log, epoch)
        writer.add_scalar("d1/Epoch"      , d1, epoch)
        writer.add_scalar("d2/Epoch"      , d2, epoch)
        writer.add_scalar("d3/Epoch"      , d3, epoch)
        # Flush events on disk
        writer.flush()


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
                original_stdout = sys.stdout
                sys.stdout = log
                summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=batch_size)
                sys.stdout = original_stdout
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
            original_stdout = sys.stdout
            sys.stdout = log
            summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=batch_size)
            sys.stdout = original_stdout
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
            original_stdout = sys.stdout
            sys.stdout = f
            summary(net, (IM_CH_IN, IM_H_IN, IM_W_IN), batch_size=batch_size)
            print("\n\n")
            printed_model = net.to('cpu')
            stat(printed_model, (IM_CH_IN, IM_H_IN, IM_W_IN))
            sys.stdout = original_stdout

    """
    DELETE CHECKPOINT FILE
    """

    if DELETE_CKPT_AFTER_TRAINING:
        print("Removing checkpoint folder after training..")
        os.remove(f"{checkpoint_folder}")
            
    # Close tensorboard
    writer.close()
