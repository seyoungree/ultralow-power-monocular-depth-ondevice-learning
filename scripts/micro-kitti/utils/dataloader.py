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

import os
import torch
import torchvision.transforms.functional as tfun
import pandas as pd
# from skimage import io
# import cv2
import PIL.Image as Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import scipy.ndimage as scpim


"""
TRANSFORM FUNCTIONS ON BOTH IMAGES AND LABELS
"""

def RotateRandom(imgL, imgR, depthGT, dispL, depthL, dispR, depthR):

    rotation = np.random.uniform(-30, 30)

    imgL    = tfun.rotate(imgL,    rotation)
    imgR    = tfun.rotate(imgR,    rotation)
    depthGT = depthGT.unsqueeze(0)
    depthGT = tfun.rotate(depthGT, rotation)
    depthGT = depthGT.squeeze(0)
    dispL   = dispL.unsqueeze(0)
    dispL   = tfun.rotate(dispL,    rotation)
    dispL   = dispL.squeeze(0)
    depthL  = depthL.unsqueeze(0)
    depthL  = tfun.rotate(depthL,   rotation)
    depthL  = depthL.squeeze(0)   
    dispR   = dispR.unsqueeze(0)
    dispR   = tfun.rotate(dispR,    rotation)
    dispR   = dispR.squeeze(0)
    depthR  = depthR.unsqueeze(0)
    depthR  = tfun.rotate(depthR,   rotation)
    depthR  = depthR.squeeze(0) 

    return imgL, imgR, depthGT, dispL, depthL, dispR, depthR

def MirrorRandom(imgL, imgR, depthGT, dispL, depthL, dispR, depthR):

    flip = np.random.randint(0, 2) # Generate a 'flip' / 'do not flip' flag

    # TO be consistent, detph maps and disparity maps, when rotated, need to be refered to
    # the rotated right image -> DISPARITY MAP IS CONSISTENT WITH RIGHT IMAGE (WITH OPPOSITE SIGN)
    if flip == 1:
        imgL_f   = tfun.hflip(imgL   ) 
        imgR_f   = tfun.hflip(imgR   )
        # Switch flipped images
        imgL     = imgR_f
        imgR     = imgL_f
        depthGT  = tfun.hflip(depthGT)   # TO BE USED WITH RESPECT TO THE RIGHT IMAGE (I.E. LEFT FLIPPED)
        dispL_f  = tfun.hflip(dispL  )
        depthL_f = tfun.hflip(depthL )
        dispR_f  = tfun.hflip(dispR  )
        depthR_f = tfun.hflip(depthR )
        # Switch flipped disparities / depths
        dispL    = dispR_f
        depthL   = depthR_f
        dispR    = dispL_f
        depthR   = depthL_f
    else:
        pass

    return imgL, imgR, depthGT, dispL, depthL, dispR, depthR

def VerticalMirrorRandom(imgL, imgR, depthGT, dispL, depthL, dispR, depthR):

    flip = np.random.randint(0, 2) # Generate a 'flip' / 'do not flip' flag

    if flip == 1:
        imgL    = tfun.vflip(imgL   )
        imgR    = tfun.vflip(imgR   )
        depthGT = tfun.vflip(depthGT)
        dispL   = tfun.vflip(dispL  )
        depthL  = tfun.vflip(depthL )
        dispR   = tfun.vflip(dispR  )
        depthR  = tfun.vflip(depthR )
    else:
        pass

    return imgL, imgR, depthGT, dispL, depthL, dispR, depthR


"""
TRANSFORM ONLY IMAGES AND NOT LABELS
"""

# Transformations from the paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9733979&casa_token=TpGmYWei0ZQAAAAA:Amcbde8vsyzpwDavQEsAwFQEwur3CefF0ljttjrX5czl-9LmwbJxlvq2FjEF4GtBcEDzsLN8mQ
def RandomGammaCorrection(imgL, imgR):
    activate = np.random.randint(0, 2) # Generate an activation flag

    if activate == 1:
        gamma = np.random.uniform(0.8, 1.2)

        imgL = tfun.adjust_gamma(imgL, gamma, 1)
        imgR = tfun.adjust_gamma(imgR, gamma, 1)
    else:
        pass

    return imgL, imgR

def ChannelwiseColorChange(imgL, imgR):
    activate = np.random.randint(0, 2) # Generate an activation flag
    color_shift_R = np.random.uniform(0.8, 1.2)
    color_shift_G = np.random.uniform(0.8, 1.2)
    color_shift_B = np.random.uniform(0.8, 1.2)

    if activate == 1:
        imgL[0, :, :] = imgL[0, :, :] * color_shift_R
        imgL[1, :, :] = imgL[1, :, :] * color_shift_G
        imgL[2, :, :] = imgL[2, :, :] * color_shift_B
        imgR[0, :, :] = imgR[0, :, :] * color_shift_R
        imgR[1, :, :] = imgR[1, :, :] * color_shift_G
        imgR[2, :, :] = imgR[2, :, :] * color_shift_B
    else:
        pass

    return imgL, imgR

def RandomBrightnessAddition(imgL, imgR):
    activate = np.random.randint(0, 2) # Generate an activation flag
    brightness = np.random.uniform(0.5, 2.0)

    if activate == 1:
        imgL = tfun.adjust_brightness(imgL, brightness)
        imgR = tfun.adjust_brightness(imgR, brightness)
    else:
        pass

    return imgL, imgR


# Other Transformations
def SwitchColorChannels(imgL, imgR):
    activate = np.random.randint(0, 2) # Generate an activation flag

    if activate == 1:
        pass
    else:
        pass

    return imgL, imgR

def AdditiveNoise(imgL, imgR):
    activate = np.random.randint(0, 2) # Generate an activation flag

    if activate == 1:
        pass
    else:
        pass

    return imgL, imgR

def RandomColor2Grayscale_KeepChannelNumber(imgL, imgR):
    activate = np.random.randint(0, 2) # Generate an activation flag

    if activate == 1:
        pass
    else:
        pass

    return imgL, imgR


"""
DATALOADERS AND DATASETS
"""

class miniKITTI(Dataset):

    def __init__(self, root_dir, transform=False, set='train', resolution='48x48', normalize=True, flip_horizontally=True, train_subset_percentage=None):
        """
        root_dir: root directory of mini-kitti
        transform: insert your transform here (custom defined here can be used)
        set: 'train', 'val', 'test'
        resolution: desired resolution here (i.e., 320x320, 48x48, or 32x32)
        normalize: set to 'True' to normalize the input images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.set = set
        self.resolution = resolution
        self.split = f'kitti_{self.resolution}/{self.set}/'
        self.normalize = normalize
        self.hflip = flip_horizontally
        if set == 'train':
            if train_subset_percentage != None and train_subset_percentage < 100 and train_subset_percentage > 0:
                self.dataset_annotations = pd.read_csv(f'{self.root_dir}/{self.split}/annotations_{train_subset_percentage}perc.csv', header=None)
            else:
                self.dataset_annotations = pd.read_csv(f'{self.root_dir}/{self.split}/annotations.csv', header=None)
        elif set == 'val':
            self.dataset_annotations = pd.read_csv(f'{self.root_dir}/{self.split}/annotations.csv', header=None)
        elif set == 'test':
            self.dataset_annotations = pd.read_csv(f'{self.root_dir}/{self.split}/annotations.csv', header=None)
        else:
            print("[miniKITTI] Invalid dataset split (not train, val, test)!!")
            exit()

    def __len__(self):
        return len(self.dataset_annotations)

    def __getitem__(self, idx):
        data = {}

        imgL_name    = os.path.join(self.root_dir, self.split, self.dataset_annotations.iloc[idx, 0])
        imgL         = Image.open(imgL_name)
        imgR_name    = os.path.join(self.root_dir, self.split, self.dataset_annotations.iloc[idx, 1])
        imgR         = Image.open(imgR_name)
        depthGT_name = os.path.join(self.root_dir, self.split, self.dataset_annotations.iloc[idx, 2])
        depthGT      = np.load(depthGT_name)
        dispL_name   = os.path.join(self.root_dir, self.split, self.dataset_annotations.iloc[idx, 3])
        dispL        = np.load(dispL_name)
        depthL_name  = os.path.join(self.root_dir, self.split, self.dataset_annotations.iloc[idx, 4])
        depthL       = np.load(depthL_name)
        dispR_name   = os.path.join(self.root_dir, self.split, self.dataset_annotations.iloc[idx, 5])
        dispR        = np.load(dispR_name)
        depthR_name  = os.path.join(self.root_dir, self.split, self.dataset_annotations.iloc[idx, 6])
        depthR       = np.load(depthR_name)
        fb_name      = os.path.join(self.root_dir, self.split, self.dataset_annotations.iloc[idx, 7])
        fb           = np.load(fb_name)

        imgL    = tfun.pil_to_tensor(imgL)
        imgR    = tfun.pil_to_tensor(imgR)
        depthGT = torch.from_numpy(depthGT)
        dispL   = torch.from_numpy(dispL)
        depthL  = torch.from_numpy(depthL)
        dispR   = torch.from_numpy(dispR)
        depthR  = torch.from_numpy(depthR)
        fb      = torch.from_numpy(fb)

        """
        TODO: insert transforms here, distinguishing which ones need to distort also the ground_truth / proxy maps
        """
        with torch.no_grad():
            if self.transform == True:
                # Transformations involving both images and label
                if self.hflip == True:
                    imgL, imgR, depthGT, dispL, depthL, dispR, depthR = MirrorRandom(imgL, imgR, depthGT, dispL, depthL, dispR, depthR)
                # imgL, imgR, depthGT, dispL, depthL, dispR, depthR = VerticalMirrorRandom(imgL, imgR, depthGT, disp, depth)
                # Transformations only on the images
                imgL, imgR = RandomGammaCorrection(imgL, imgR)
                imgL, imgR = ChannelwiseColorChange(imgL, imgR)
                imgL, imgR = RandomBrightnessAddition(imgL, imgR)
            if self.normalize == True:
                imgL = imgL.float() / 255
                imgR = imgR.float() / 255
                
        # Pack data

        data['imgL']    = imgL
        data['imgR']    = imgR
        data['depthGT'] = depthGT
        data['dispL']   = dispL
        data['depthL']  = depthL
        data['dispR']   = dispR
        data['depthR']  = depthR
        data['fb']      = fb
        
        return data
    
    def getstats(self, idx):
        metrics_name = os.path.join(self.root_dir, self.split, self.dataset_annotations.iloc[idx, 8])
        metrics      = pd.read_csv(metrics_name)

        return metrics

    def getimagesize(self):
        img_name = os.path.join(self.root_dir, self.split, self.dataset_annotations.iloc[0, 0])
        image = Image.open(img_name)
        image = tfun.pil_to_tensor(image)
        size = image.shape

        return size
    
    def getdepthsize(self):
        depth_name = os.path.join(self.root_dir, self.split, self.dataset_annotations.iloc[0, 4])
        depth = np.load(depth_name)
        size = depth.shape

        return size


""" TEST DATALOADER """
if __name__ == '__main__':

    main_dir   = "/home/pulp/mini-kitti/"

    transform  = True

    dataset    = miniKITTI(root_dir=f'{main_dir}', transform=transform, set='train', resolution='48x48', normalize=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    for i in range(len(dataset)):

        fig, ax = plt.subplots(3, 3)

        data = dataset[i]

        ax[0][0].imshow(data['imgL'].permute(1,2,0).numpy())
        ax[0][1].imshow(data['imgR'].permute(1,2,0).numpy())
        ax[0][2].imshow(data['depthGT'])
        ax[1][0].imshow(data['dispL'])
        ax[2][0].imshow(data['depthL'])
        ax[1][1].imshow(data['dispR'])
        ax[2][1].imshow(data['depthR'])
        
        ax[0][0].title.set_text("imgL")
        ax[0][1].title.set_text("imgR")
        ax[0][2].title.set_text("depthGT")
        ax[1][0].title.set_text("dispL")
        ax[2][0].title.set_text("depthL")
        ax[1][1].title.set_text("dispR")
        ax[2][1].title.set_text("depthR")

        print(f"fb product is {data['fb']}")

        plt.show()

        if i == 3:
            break
