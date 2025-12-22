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
References: 
- https://paperswithcode.com/dataset/tartanair
- https://github.com/castacks/tartanair_tools/blob/master/download_training_zipfiles.txt
- https://www.aicrowd.com/challenges/tartanair-visual-slam-mono-track
"""

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

def RotateRandom(imgL, depthL, dispL):

    rotation = np.random.uniform(-30, 30)

    imgL    = tfun.rotate(imgL,    rotation)
    dispL   = dispL.unsqueeze(0)
    dispL   = tfun.rotate(dispL,    rotation)
    dispL   = dispL.squeeze(0)
    depthL  = depthL.unsqueeze(0)
    depthL  = tfun.rotate(depthL,   rotation)
    depthL  = depthL.squeeze(0)   

    return imgL, depthL, dispL

def MirrorRandom(imgL, depthGT, depthL, dispL):

    flip = np.random.randint(0, 2) # Generate a 'flip' / 'do not flip' flag

    # Here, since the stereo couple is not present, disparity is arbitrarily refered to a fake right image
    if flip == 1:
        imgL_f    = tfun.hflip(imgL   )
        # Switch flipped images
        imgL      = imgL_f
        dispL_f   = tfun.hflip(dispL  )
        depthL_f  = tfun.hflip(depthL )
        depthGT_f = tfun.hflip(depthGT)
        # Switch flipped disparities / depths
        dispL     = dispL_f
        depthL    = depthL_f
        depthGT   = depthGT_f
    else:
        pass

    return imgL, depthGT, depthL, dispL

def VerticalMirrorRandom(imgL, depthL, dispL):

    flip = np.random.randint(0, 2) # Generate a 'flip' / 'do not flip' flag

    if flip == 1:
        imgL    = tfun.vflip(imgL   )
        dispL   = tfun.vflip(dispL  )
        depthL  = tfun.vflip(depthL )
    else:
        pass

    return imgL, depthL, dispL


"""
TRANSFORM ONLY IMAGES AND NOT LABELS
"""

# Transformations from the paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9733979&casa_token=TpGmYWei0ZQAAAAA:Amcbde8vsyzpwDavQEsAwFQEwur3CefF0ljttjrX5czl-9LmwbJxlvq2FjEF4GtBcEDzsLN8mQ
def RandomGammaCorrection(imgL):
    activate = np.random.randint(0, 2) # Generate an activation flag

    if activate == 1:
        gamma = np.random.uniform(0.8, 1.2)

        imgL = tfun.adjust_gamma(imgL, gamma, 1)
    else:
        pass

    return imgL

def ChannelwiseColorChange(imgL):
    activate = np.random.randint(0, 2) # Generate an activation flag
    color_shift_R = np.random.uniform(0.8, 1.2)
    color_shift_G = np.random.uniform(0.8, 1.2)
    color_shift_B = np.random.uniform(0.8, 1.2)

    if activate == 1:
        imgL[0, :, :] = imgL[0, :, :] * color_shift_R
        imgL[1, :, :] = imgL[1, :, :] * color_shift_G
        imgL[2, :, :] = imgL[2, :, :] * color_shift_B
    else:
        pass

    return imgL

def RandomBrightnessAddition(imgL):
    activate = np.random.randint(0, 2) # Generate an activation flag
    brightness = np.random.uniform(0.5, 2.0)

    if activate == 1:
        imgL = tfun.adjust_brightness(imgL, brightness)
    else:
        pass

    return imgL

"""
TRANSFORMS ON THE DEPTH / DISPARITY MAP (ADJUSTMENTS)
"""

def clip_depth_to_max_value(depthL, MAX_DEPTH=80.0):

    invalid_values = torch.zeros_like(depthL)
    invalid_values = (depthL > MAX_DEPTH)

    depthL[invalid_values] = MAX_DEPTH

    return depthL

# FB product is 80, so MIN_DISP = 1
def clip_disparity_to_min_value(dispL, MIN_DISP=1.0):

    invalid_values = torch.zeros_like(dispL)
    invalid_values = (dispL < MIN_DISP)

    dispL[invalid_values] = MIN_DISP

    return dispL



"""
DATALOADERS AND DATASETS
"""

class miniNYUv2(Dataset):

    def __init__(self, root_dir, transform=False, set='train', normalize=True, flip_horizontally=True, train_subset_percentage=None, test_subset_percentage=None):
        """
        root_dir: root directory of mini-kitti
        transform: insert your transform here (custom defined here can be used)
        set: 'train', 'val', 'test'
        normalize: set to 'True' to normalize the input images
        flip_horizontally: set to 'True' for flipping at training time
        """
        self.root_dir = root_dir
        self.transform = transform
        self.set = set
        self.normalize = normalize
        self.hflip = flip_horizontally
        if set == 'train':
            if train_subset_percentage != None and train_subset_percentage < 100 and train_subset_percentage > 0:
                self.dataset_annotations = pd.read_csv(f'{self.root_dir}/{self.set}_{train_subset_percentage}perc.csv', header=None)
            else:
                self.dataset_annotations = pd.read_csv(f'{self.root_dir}/{self.set}.csv', header=None)
        elif set == 'val':
            self.dataset_annotations = pd.read_csv(f'{self.root_dir}/{self.set}.csv', header=None)
        elif set == 'test':
            if test_subset_percentage != None and test_subset_percentage < 100 and test_subset_percentage > 0:
                self.dataset_annotations = pd.read_csv(f'{self.root_dir}/{self.set}_{test_subset_percentage}perc.csv', header=None)
            else:
                self.dataset_annotations = pd.read_csv(f'{self.root_dir}/{self.set}.csv', header=None)
        else:
            print("[miniNYUv2] Invalid dataset split (not train, val, test)!!")
            exit()

    def __len__(self):
        return len(self.dataset_annotations)

    def __getitem__(self, idx):
        data = {}

        imgL_name    = os.path.join(self.root_dir, self.dataset_annotations.iloc[idx, 0])
        imgL         = Image.open(imgL_name)
        depthGT_name = os.path.join(self.root_dir, self.dataset_annotations.iloc[idx, 1])
        depthGT      = np.load(depthGT_name)       
        depthL_name  = os.path.join(self.root_dir, self.dataset_annotations.iloc[idx, 2])
        depthL       = np.load(depthL_name)
        dispL_name   = os.path.join(self.root_dir, self.dataset_annotations.iloc[idx, 3])
        dispL        = np.load(dispL_name)
        fb_name      = os.path.join(self.root_dir, self.dataset_annotations.iloc[idx, 4])
        fb           = np.load(fb_name)

        imgL    = tfun.pil_to_tensor(imgL)
        depthGT = torch.from_numpy(depthGT)
        depthL  = torch.from_numpy(depthL)
        dispL   = torch.from_numpy(dispL)
        fb      = torch.from_numpy(fb)

        """
        TODO: insert transforms here, distinguishing which ones need to distort also the ground_truth / proxy maps
        """
        with torch.no_grad():
            if self.transform == True:
                # Transformations involving both images and label
                if self.hflip == True:
                    imgL, depthGT, depthL, dispL = MirrorRandom(imgL, depthGT, depthL, dispL)
                # Transformations only on the images
                imgL = RandomGammaCorrection(imgL)
                imgL = ChannelwiseColorChange(imgL)
                imgL = RandomBrightnessAddition(imgL)
            if self.normalize == True:
                imgL = imgL.float() / 255
            # Set max depth / min disparity into depth map
            depthL = clip_depth_to_max_value(depthL, 5.0)
            dispL  = clip_disparity_to_min_value(dispL, 1.0)
                
        # Pack data
        data['imgL']    = imgL
        data['depthGT'] = depthGT
        data['depthL']  = depthL
        data['dispL']   = dispL
        data['fb']      = fb
        
        return data

    def getimagesize(self):
        img_name = os.path.join(self.root_dir, self.dataset_annotations.iloc[0, 0])
        image = Image.open(img_name)
        image = tfun.pil_to_tensor(image)
        size = image.shape

        return size
    
    def getdepthsize(self):
        depth_name = os.path.join(self.root_dir, self.dataset_annotations.iloc[0, 2])
        depth = np.load(depth_name)
        size = depth.shape

        return size


""" TEST DATALOADER """
if __name__ == '__main__':

    main_dir   = "/home/pulp/mini-nyuv2/"

    transform  = False
    hflip = True
    use_dataloader = True

    dataset    = miniNYUv2(root_dir=f'{main_dir}', transform=transform, set='val', normalize=True, flip_horizontally=hflip)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    print(f"Dataset is {len(dataset)} samples long.")

    i = 0
    for j in range(int(len(dataset)/3)):

        fig, ax = plt.subplots(4, 3)

        if use_dataloader:
            data = next(iter(dataloader))

            ax[0][0].imshow(data['imgL'][0].permute(1,2,0).numpy())
            ax[1][0].imshow(data['depthL'][0])
            ax[2][0].imshow(data['dispL'][0])
            ax[3][0].imshow(data['depthGT'][0])
            ax[0][1].imshow(data['imgL'][1].permute(1,2,0).numpy())
            ax[1][1].imshow(data['depthL'][1])
            ax[2][1].imshow(data['dispL'][1])
            ax[3][1].imshow(data['depthGT'][1])
            ax[0][2].imshow(data['imgL'][2].permute(1,2,0).numpy())
            ax[1][2].imshow(data['depthL'][2])
            ax[2][2].imshow(data['dispL'][2])
            ax[3][2].imshow(data['depthGT'][2])
            
            ax[0][0].title.set_text("imgL")
            ax[1][0].title.set_text("depthL")
            ax[2][0].title.set_text("dispL")
            ax[3][0].title.set_text("depthGT")
            ax[0][1].title.set_text("imgL")
            ax[1][1].title.set_text("depthL")
            ax[2][1].title.set_text("dispL")
            ax[3][1].title.set_text("depthGT")
            ax[0][2].title.set_text("imgL")
            ax[1][2].title.set_text("depthL")
            ax[2][2].title.set_text("dispL")
            ax[3][2].title.set_text("depthGT")

            print(f"fb product is {data['fb'][0]}")

            plt.show()

        else:
            data_0 = dataset[i]
            data_1 = dataset[i+1]
            data_2 = dataset[i+2]

            ax[0][0].imshow(data_0['imgL'].permute(1,2,0).numpy())
            ax[1][0].imshow(data_0['depthL'])
            ax[2][0].imshow(data_0['dispL'])
            ax[3][0].imshow(data_0['depthGT'])
            ax[0][1].imshow(data_1['imgL'].permute(1,2,0).numpy())
            ax[1][1].imshow(data_1['depthL'])
            ax[2][1].imshow(data_1['dispL'])
            ax[3][1].imshow(data_1['depthGT'])
            ax[0][2].imshow(data_2['imgL'].permute(1,2,0).numpy())
            ax[1][2].imshow(data_2['depthL'])
            ax[2][2].imshow(data_2['dispL'])
            ax[3][2].imshow(data_2['depthGT'])
            
            ax[0][0].title.set_text("imgL")
            ax[1][0].title.set_text("depthL")
            ax[2][0].title.set_text("dispL")
            ax[3][0].title.set_text("depthGT")
            ax[0][1].title.set_text("imgL")
            ax[1][1].title.set_text("depthL")
            ax[2][1].title.set_text("dispL")
            ax[3][1].title.set_text("depthGT")
            ax[0][2].title.set_text("imgL")
            ax[1][2].title.set_text("depthL")
            ax[2][2].title.set_text("dispL")
            ax[3][2].title.set_text("depthGT")

            print(f"fb product is {data_0['fb']}")

            plt.show()

        i = i + 3
        if i >= 21:
            break
