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
SOURCE: https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Metrics using torch
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths (KITTI)
    """
    # From https://github.com/jaehanlee-mcl/monocular-depth-estimation-using-relative-depth-maps/blob/master/evaluation.m
    tmap = torch.maximum( pred/gt, gt/pred )
    c1   = tmap < 1.25
    c2   = tmap < 1.25 ** 2
    c3   = tmap < 1.25 ** 3
    a1 = torch.mean( c1.float() )
    a2 = torch.mean( c2.float() )
    a3 = torch.mean( c3.float() )   
    
    # abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / gt_valid)
    data_size = pred.numel()
    abs_rel   = ( torch.sum((pred - gt) / gt) ) / data_size

    # sq_rel = torch.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    sq_rel = torch.sum(((gt - pred) ** 2) / gt) / data_size

    # rmse = (gt_valid - pred_valid) ** 2
    # rmse = torch.sqrt(rmse.mean())
    rmse = torch.sum((gt - pred) ** 2) / data_size
    rmse = torch.sqrt(rmse)

    # rmse_log = (torch.log(gt_valid) - torch.log(pred_valid)) ** 2
    # rmse_log = torch.sqrt(rmse_log.mean())
    rmse_log = torch.sum((torch.log(gt) - torch.log(pred)) ** 2) / data_size
    rmse_log = torch.sqrt(rmse_log)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

# Metrics using torch and with LIDAR ground truth masking (FOR TRAINING)
def compute_errors_masked_train(gt, pred):
    """Computation of error metrics between predicted and ground truth depths (KITTI)
    """

    # Step 1: find mask to compute metrics only on valid values
    gt_mask   = (gt > 0) 
    pred_mask = (pred > 0) #(pred != -1)
    vmask     = torch.logical_and(gt_mask, pred_mask)

    # Step 2: collect valid values on dense arrays (both gt and pred)
    gt_valid    = gt[vmask]
    pred_valid  = pred[vmask]

    # Step 3: compute and return metrics
    # From https://github.com/jaehanlee-mcl/monocular-depth-estimation-using-relative-depth-maps/blob/master/evaluation.m
    tmap = torch.maximum( pred_valid/gt_valid, gt_valid/pred_valid )
    c1   = tmap < 1.25
    c2   = tmap < 1.25 ** 2
    c3   = tmap < 1.25 ** 3
    a1 = torch.mean( c1.float() )
    a2 = torch.mean( c2.float() )
    a3 = torch.mean( c3.float() )  

    data_size = pred_valid.numel()

    # abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / gt_valid)
    abs_rel   = torch.sum( torch.abs(pred_valid - gt_valid) / gt_valid) / data_size

    # sq_rel = torch.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    sq_rel = torch.sum(((gt_valid - pred_valid) ** 2) / gt_valid) / data_size

    # rmse = (gt_valid - pred_valid) ** 2
    # rmse = torch.sqrt(rmse.mean())
    rmse = torch.sum((gt_valid - pred_valid) ** 2) / data_size
    rmse = torch.sqrt(rmse)

    rmse_log = (torch.log(gt_valid) - torch.log(pred_valid)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    # rmse_log = torch.sum((torch.log(gt_valid) - torch.log(pred_valid)) ** 2) / data_size
    # rmse_log = torch.sqrt(rmse_log)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# Metrics using torch and with LIDAR ground truth masking (FOR TRAINING)
def compute_errors_masked_train_max_dpth(gt, pred, max_dpth):
    """Computation of error metrics between predicted and ground truth depths (KITTI)
    """

    # Step 1: find mask to compute metrics only on valid values
    gt_mask       = (gt > 0)
    gt_mask_max   = (gt <= max_dpth)
    pred_mask     = (pred > 0) #(pred != -1)
    pred_mask_max = (pred <= max_dpth)
    distmask      = torch.logical_and(gt_mask_max, pred_mask_max)
    vmask         = torch.logical_and(gt_mask, pred_mask)
    vmask         = torch.logical_and(vmask, distmask)

    # Step 2: collect valid values on dense arrays (both gt and pred)
    gt_valid    = gt[vmask]
    pred_valid  = pred[vmask]

    # Step 3: compute and return metrics
    # From https://github.com/jaehanlee-mcl/monocular-depth-estimation-using-relative-depth-maps/blob/master/evaluation.m
    tmap = torch.maximum( pred_valid/gt_valid, gt_valid/pred_valid )
    c1   = tmap < 1.25
    c2   = tmap < 1.25 ** 2
    c3   = tmap < 1.25 ** 3
    a1 = torch.mean( c1.float() )
    a2 = torch.mean( c2.float() )
    a3 = torch.mean( c3.float() )  

    data_size = pred_valid.numel()

    # abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / gt_valid)
    abs_rel   = torch.sum( torch.abs(pred_valid - gt_valid) / gt_valid) / data_size

    # sq_rel = torch.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    sq_rel = torch.sum(((gt_valid - pred_valid) ** 2) / gt_valid) / data_size

    # rmse = (gt_valid - pred_valid) ** 2
    # rmse = torch.sqrt(rmse.mean())
    rmse = torch.sum((gt_valid - pred_valid) ** 2) / data_size
    rmse = torch.sqrt(rmse)

    rmse_log = (torch.log(gt_valid) - torch.log(pred_valid)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    # rmse_log = torch.sum((torch.log(gt_valid) - torch.log(pred_valid)) ** 2) / data_size
    # rmse_log = torch.sqrt(rmse_log)
    
    #import pdb; pdb.set_trace()

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# Metrics using torch and proxy disparity maps for INFERENCE
def compute_errors_masked_inference(pr_gt, pred):
    """Computation of error metrics between predicted and ground truth depths (KITTI)
    """

    pr_gt = pr_gt.squeeze(0)

    # Step 1: find mask to compute metrics only on valid values
    gt_mask   = (pr_gt > 0) 
    pred_mask = (pred > 0)
    neg_pred  = (pred < 0)
    vmask     = torch.logical_and(gt_mask, pred_mask)

    negative_predictions = torch.sum(neg_pred)

    # Step 2: collect valid values on dense arrays (both gt and pred)
    gt_valid    = pr_gt[vmask]
    pred_valid  = pred[vmask]

    # Step 3: compute and return metrics
    # From https://github.com/jaehanlee-mcl/monocular-depth-estimation-using-relative-depth-maps/blob/master/evaluation.m
    tmap = torch.maximum( pred_valid/gt_valid, gt_valid/pred_valid )
    c1   = tmap < 1.25
    c2   = tmap < 1.25 ** 2
    c3   = tmap < 1.25 ** 3
    a1 = torch.mean( c1.float() )
    a2 = torch.mean( c2.float() )
    a3 = torch.mean( c3.float() )  

    data_size = pred_valid.numel()
    
    abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / gt_valid)
    # abs_rel   = ( torch.sum((pred_valid - gt_valid) / gt_valid) ) / data_size

    # sq_rel = torch.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    sq_rel = torch.sum(((gt_valid - pred_valid) ** 2) / gt_valid) / data_size

    # rmse = (gt_valid - pred_valid) ** 2
    # rmse = torch.sqrt(rmse.mean())
    rmse = torch.sum((gt_valid - pred_valid) ** 2) / data_size
    rmse = torch.sqrt(rmse)

    # rmse_log = (torch.log(gt_valid) - torch.log(pred_valid)) ** 2
    # rmse_log = torch.sqrt(rmse_log.mean())
    rmse_log = torch.sum((torch.log(gt_valid) - torch.log(pred_valid)) ** 2) / data_size
    rmse_log = torch.sqrt(rmse_log)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, negative_predictions



"""
TEST FUNCTIONS
"""
if __name__ == '__main__':

    data = torch.tensor([[3.5, 4, 0,   1],
                         [6,   0, 1.8, 2]])

    label = torch.tensor([[2.5, 3.5, 0, 1.2],
                          [3.4, -1, -1, 1.6]])
    
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors_masked_train(label, data)

    print(f"abs_rel = {abs_rel:.3f}, sq_rel = {sq_rel:.3f}, rmse = {rmse:.3f}, rmse_log = {rmse_log:.3f}, a1 = {a1:.3f}, a2 = {a2:.3f}, a3 = {a3:.3f}")

