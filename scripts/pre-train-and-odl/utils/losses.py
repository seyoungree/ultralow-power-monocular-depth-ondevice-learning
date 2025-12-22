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
SOURCE: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9422776
RE-PROJECTION LOSS: https://github.com/mrharicot/monodepth/blob/b76bee4bd12610b482163871b7ff93e931cb5331/monodepth_model.py#L334
'''

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from ignite.metrics import SSIM
from .processing import *

# berHu loss computed for each element and used for this loss
# SOURCE: https://github.com/abduallahmohamed/reversehuberloss/blob/master/rhuloss.py
def ProxySupervisionLoss(output, target, alpha=0.2, invalid_value=-1, device='cpu', data_type='fp32'):

    # First, find the mask of valid values
    valid_mask = (target != invalid_value) # (target > invalid_value).float()
    if data_type == 'fp32':
        N_valid = torch.sum(valid_mask).float()
    elif data_type == 'fp16':
        N_valid = torch.sum(valid_mask).half()
    elif data_type == 'bfloat16':
        N_valid = torch.sum(valid_mask).bfloat16()

    # Then, normalize prediction and target according to the max value of target disparity
    norm_value = torch.max(target)
    target     = target / norm_value
    output     = output / norm_value

    if output.size() != target.size():
        raise Exception("[ProxySuperVisionLoss] Differently sized output and target!")

    # Compute the berHu loss on the normalized and masked values
    absdiff   = torch.abs(output - target)
    C         = 0.2 * torch.max(absdiff).item()
    berhu_map = torch.where(absdiff < C, absdiff, (absdiff*absdiff - C*C) / (2*C))
    valid_map = torch.zeros_like(output).to(device)
    valid_map[valid_mask] = berhu_map[valid_mask]
    loss      = torch.sum(valid_map) / N_valid

    return loss

# berHu loss computed for each element and used for this loss
# SOURCE: https://github.com/abduallahmohamed/reversehuberloss/blob/master/rhuloss.py
def Test_ProxySupervisionLoss(output, target, alpha=0.2, invalid_value=-1, device='cpu', data_type='fp32'):

    # First, find the mask of valid values
    valid_mask = (target != invalid_value) # (target > invalid_value).float()
    if data_type == 'fp32':
        N_valid = torch.sum(valid_mask).float()
        dttype = torch.torch.FloatTensor
    elif data_type == 'fp16':
        N_valid = torch.sum(valid_mask).half()
        dttype = torch.torch.HalfTensor
    elif data_type == 'bfloat16':
        N_valid = torch.sum(valid_mask).bfloat16()
        dttype = torch.torch.BFloat16Tensor

    # Then, normalize prediction and target according to the max value of target disparity
    norm_value = torch.max(target)
    target     = target / norm_value
    output     = output / norm_value

    if output.size() != target.size():
        raise Exception("[ProxySuperVisionLoss] Differently sized output and target!")

    # Compute the berHu loss on the normalized and masked values
    absdiff   = torch.abs(output - target)
    C         = 0.2 * torch.max(absdiff).item()
    berhu_map = torch.where(absdiff < C, absdiff, (absdiff*absdiff - C*C) / (2*C))
    valid_map = torch.zeros_like(output).type(dttype).to(device)
    valid_map[valid_mask] = berhu_map[valid_mask].type(dttype).to(device)
    loss      = torch.sum(valid_map) / N_valid

    return loss

# Photometric Reprojection Loss as defined in the formula (equivalent to the on of Monodepth2)
def PhotometricReprojectionLoss(imgL, imgR, disparity, device, data_type='fp32'):

    batch_size, ch, hin, win = imgL.size()

    warped_R = torch.zeros_like(imgL)
    for smpl in range(batch_size):
        imgR1 = imgR[smpl].unsqueeze(0)
        disp1 = disparity[smpl].unsqueeze(0).unsqueeze(0)
        if data_type == 'fp32':
            w_R = warp_right_with_disparity(imgR1, -disp1, device).float()
        elif data_type == 'fp16':
            w_R = warp_right_with_disparity(imgR1, -disp1, device).half()
        elif data_type == 'bfloat16':
            w_R = warp_right_with_disparity(imgR1, -disp1, device).bfloat16()
        w_R = w_R.squeeze(0)
        warped_R[smpl] = w_R

    # figx, axx = plt.subplots(2,1)
    # for smpl in range(batch_size):
    #     imgL_np = imgL[smpl].permute(1,2,0).to('cpu').numpy()
    #     wR_np   = warped_R[smpl].permute(1,2,0).to('cpu').numpy()
    #     imgL_np = convert_img_float_to_uint8(imgL_np)
    #     wR_np   = convert_img_float_to_uint8(wR_np)
    #     axx[0].imshow(imgL_np)
    #     axx[1].imshow(wR_np)
    #     plt.show()

    ssim_L = compute_ssim_ignite(imgL, warped_R)
    mod_LwR = torch.abs(imgL - warped_R)
    mod_LwR = mod_LwR.mean() 
    
    La = (1 - ssim_L) / 2
    Lb = mod_LwR
    Lap = 0.85 * La + 0.15 * Lb

    return Lap, La, Lb


# Sum of previous losses, this computes:
# Linit = aap * Lap + aps * Lps
# where Lap is the ProtometricReprojectionLoss and Lps is the ProxySupervisionLoss
def MonocularDepthSemiSupervisedLoss(output, target, imgL, imgR, aap, aps, alpha, invalid_value, original_width, device, data_type='fp32'):
    """
    Parameters:
    output:         output of the network (estimated disparity map)
    target:         proxy-label from SGM (noisy disparity map)
    imgL:           input image from which the model predicts monodepth
    imgR:           right image of the stereo input (only used in training)
    aap:            multiplicative coefficient for the protometric reprojection loss 
    aps:            multiplicative coefficient for the proxy-supervision loss 
    alpha:          berHu tuning parameter (usually 0.2)
    invalid_value:  value of the invalid pixels in the proxy-supervision label
    original_width: maximum size of the mini-KITTI images (from which the tiny images are downsampled)
    device:         device used by pytorch to compute ('cpu' or 'cuda')
    """

    Lap, La, Lb = PhotometricReprojectionLoss(imgL, imgR, output, device, data_type)
    Lps         = ProxySupervisionLoss       (output, target, alpha, invalid_value, device, data_type)

    Lmonodept = (aap * Lap) + (aps * Lps)

    return Lmonodept, Lap, Lps, La, Lb 



"""
OTHER LOSSES USED IN MONOCULAR DEPTH ESTIMATION
"""

# SOURCE: https://github.com/dg-enlens/banet-depth-prediction/blob/master/loss.py
def ScaleInvariantMSELogLoss(output, target, variance_focus: float = 0.85) -> float:
    """
    Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
    more information about scale-invariant loss.

    Args:
        output (Tensor): Prediction.
        target (Tensor): Target.
        variance_focus (float): Variance focus for the SILog computation.

    Returns:
        float: SILog loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    non_zero_mask = (target > 0) & (output > 0)

    # Normalize output and label 
    norm_value = torch.max(target)
    target     = target / norm_value
    output     = output / norm_value    

    # SILog
    d = torch.log(output[non_zero_mask]) - torch.log(target[non_zero_mask])
    return torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0

# SOURCE: https://github.com/dg-enlens/banet-depth-prediction/blob/master/loss.py
def ScaleInvariantMSELogLoss_max_dpth(output, target, variance_focus: float = 0.85, max_dpth = 80.0) -> float:
    """
    Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
    more information about scale-invariant loss.

    Args:
        output (Tensor): Prediction.
        target (Tensor): Target.
        variance_focus (float): Variance focus for the SILog computation.

    Returns:
        float: SILog loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    non_zero_mask = (target > 0) & (output > 0) & (target <= max_dpth)

    # Normalize output and label 
    norm_value = torch.max(target)
    target     = target / norm_value
    output     = output / norm_value    

    # SILog
    d = torch.log(output[non_zero_mask]) - torch.log(target[non_zero_mask])
    return torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0


def masked_MSELoss (output, target, invalid_value=-1):

    mask = (target != invalid_value).float()
    valid_entries = torch.sum(mask)

    mse_loss = F.mse_loss(output, target, reduction='none')
    masked_loss = mse_loss * mask

    if torch.sum(torch.isnan(masked_loss)) > 0:
        print('NaN values present in loss!')
        print(masked_loss)

    loss = torch.sum(masked_loss) / valid_entries

    return loss


def masked_L1Loss (output, target, invalid_value=-1):

    mask = (target != invalid_value).float()
    valid_entries = torch.sum(mask)

    mse_loss = F.l1_loss(output, target, reduction='none')
    masked_loss = mse_loss * mask

    if torch.sum(torch.isnan(masked_loss)) > 0:
        print('NaN values present in loss!')
        print(masked_loss)

    loss = torch.sum(masked_loss) / valid_entries

    return loss


"""
OLD AND MALFUNCTIONING
"""

# # berHu loss computed for each element and used for this loss
# def ProxySupervisionLoss_OLD(output, target, alpha=0.2, invalid_value=-1):
#     loss = 0
#     size = output.size()
#     dim  = output.dim()

#     # Find mask of valid values
#     valid_mask_leq = (target != invalid_value).float() # (target > invalid_value).float()
#     valid_mask_gr  = (target != invalid_value).float() # (target > invalid_value).float()
#     N_valid = torch.sum(valid_mask_leq)

#     if output.size() != target.size():
#         raise Exception("[ProxySuperVisionLoss] Differently sized output and target!")

#     # Loss computation
#     # Case of batch size > 1
#     if dim == 4:

#         N_elem = size[0] * size[1] * size[2] * size[3]
#         abs_diff = torch.abs(output - target)
#         c = alpha * (torch.max(abs_diff))

#         # Prepare elements to be multiplied for each case
#         mask_leq = abs_diff <= c
#         mask_gr  = abs_diff > c

#         # Multiply elements according to their condition (berHu)
#         loss_leq_elementwise = mask_leq * abs_diff 
#         loss_gr_elementwise  = (mask_gr  * (abs_diff*abs_diff) - c*c) / (2*c)

#         # Mask results 
#         loss_leq_elementwise_masked = loss_leq_elementwise * valid_mask_leq
#         loss_gr_elementwise_masked  = loss_gr_elementwise  * valid_mask_gr

#         # Sum and average all elements to obtain the loss
#         #loss = (torch.sum(loss_leq_elementwise) + torch.sum(loss_gr_elementwise)) / N_elem
#         loss = (torch.sum(loss_leq_elementwise_masked) + torch.sum(loss_gr_elementwise_masked)) / N_valid
    
#     # Case of batch size = 1
#     elif dim == 3:

#         N_elem = size[0] * size[1] * size[2]
#         abs_diff = torch.abs(output - target)
#         c = alpha * (torch.max(abs_diff))

#         # Prepare elements to be multiplied for each case
#         mask_leq = abs_diff <= c
#         mask_gr  = abs_diff > c

#         # Multiply elements according to their condition (berHu)
#         loss_leq_elementwise = mask_leq * abs_diff 
#         loss_gr_elementwise  = (mask_gr  * (abs_diff*abs_diff) - c*c) / (2*c)

#         # Mask results 
#         loss_leq_elementwise_masked = loss_leq_elementwise * valid_mask_leq
#         loss_gr_elementwise_masked  = loss_gr_elementwise  * valid_mask_gr

#         # Sum and average all elements to obtain the loss
#         #loss = (torch.sum(loss_leq_elementwise) + torch.sum(loss_gr_elementwise)) / N_elem
#         loss = (torch.sum(loss_leq_elementwise_masked) + torch.sum(loss_gr_elementwise_masked)) / N_valid
    
#     # Throw exception
#     else:
#         raise Exception(f"[ProxySupervisionLoss] Invalid output number of dimensions (have {dim}, should be at least 3)")

#     return loss

# # Self-supervision photometric loss as defined in uPyD-Net paper
# # Warping images: https://discuss.pytorch.org/t/warping-images-using-disparity-maps-for-stereo-matching/127234
# def PhotometricReprojectionLoss_OLD(disparity, imgL, imgR, original_width, device):

#     if imgL.dim() == 3 or imgR.dim() == 3:
#         imgL = imgL.unsqueeze(0)
#         imgR = imgR.unsqueeze(0)

#     # Warp imgR using the output disparity map to obtain the estimated imgL
#     warpedR = warp_right_with_disparity_on_downsampled(imgR.float(), -disparity, original_width, device)

#     # warpedR is [batch, ch, ??, h, w]

#     # single_imgL = imgL[4,:,:,:]
#     # single_imgL = single_imgL.permute(1,2,0).cpu().numpy()
#     # single_imgL = convert_img_float_to_uint8(single_imgL)
#     # single_imgL = single_imgL

#     # img = warpedR.permute(0,2,1,3,4)
#     # img = img[4,:,:,:,:]
#     # single_img = img[0,:,:,:]
#     # single_img = single_img.permute(1,2,0)
#     # single_img1 = img[1,:,:,:]
#     # single_img1 = single_img1.permute(1,2,0)
#     # single_img2 = img[2,:,:,:]
#     # single_img2 = single_img2.permute(1,2,0)
#     # single_img3 = img[3,:,:,:]
#     # single_img3 = single_img3.permute(1,2,0)

#     # single_img = single_img.cpu().numpy()
#     # single_img = convert_img_float_to_uint8(single_img)
#     # single_img1 = single_img1.cpu().numpy()
#     # single_img1 = convert_img_float_to_uint8(single_img1)
#     # single_img2 = single_img2.cpu().numpy()
#     # single_img2 = convert_img_float_to_uint8(single_img2)
#     # single_img3 = single_img3.cpu().numpy()
#     # single_img3 = convert_img_float_to_uint8(single_img3)

#     # fig, ax = plt.subplots(1,5)
#     # ax[0].imshow(single_imgL)
#     # ax[1].imshow(single_img)
#     # ax[2].imshow(single_img1)
#     # ax[3].imshow(single_img2)
#     # ax[4].imshow(single_img3)
#     # plt.show()

#     # Since warpedR is [batch, ch, ??, h, w], eliminate the ?? dimension
#     warpedR = warpedR.permute(2, 0, 1, 3, 4)
#     warpedR = warpedR[0, :, :, :, :]

#     # Compute the SSIM over imgL and warped imgR
#     # SOURCE: https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html
#     ssim_L = compute_ssim_ignite(imgL, warpedR)

#     # Compute the L1 loss between imgL and warped imgR
#     mod_LwR = torch.mean(torch.abs(imgL - warpedR))

#     # Put pieces together: Lap = 0.85 * (1 - ssim_L) / 2 + 0.15 * mod_LwR
#     Lap = 0.85 * (1 - ssim_L) / 2 + 0.15 * mod_LwR

#     return Lap

# # Scale-Invariant MSE Loss in log space
# def ScaleInvariantMSELogLoss(output, target, reduction=True):
#     loss = 0
#     size = output.size()
#     dim  = output.dim()

#     if output.size() != target.size():
#         raise Exception("[ScaleInvariantMSELogLoss] Differently sized output and target!")

#     # Loss computation
#     # Case of batch size > 1
#     if dim == 4:
#         N_elems = size[0] * size[1] * size[2] * size[3]

#         log_out = torch.log(output)
#         log_tgt = torch.log(target)

#         log_out = torch.nan_to_num(log_out, nan=1e-4, posinf=4.0, neginf=1e-4)
#         log_tgt = torch.nan_to_num(log_tgt, nan=1e-4, posinf=4.0, neginf=1e-4)

#         d       = log_out - log_tgt

#         # Compute loss
#         loss = (1/N_elems) * (torch.sum(d*d)) + (1/(N_elems*N_elems)) * (torch.sum(d)*torch.sum(d)) 

#     # Case of batch size = 1
#     elif dim == 3:
#         N_elems = size[0] * size[1] * size[2]

#         log_out = torch.log(output)
#         log_tgt = torch.log(target)
#         d       = log_out - log_tgt

#         # Compute loss
#         loss = (1/N_elems) * (torch.sum(d*d)) + (1/(N_elems*N_elems)) * (torch.sum(d)*torch.sum(d)) 

#     # Throw exception
#     else:
#         raise Exception(f"[ScaleInvariantMSELogLoss] Invalid output number of dimensions (have {dim}, should be at least 3)")

#     return loss


# def MaskedSILogLoss(output, target, invalid_value=-1):

#     mask = (target != invalid_value).float()
#     valid_entries = torch.sum(mask)

#     silog_loss  = ScaleInvariantMSELogLoss(output, target, reduction=False)
#     masked_loss = silog_loss * mask

#     if torch.sum(torch.isnan(masked_loss)) > 0:
#         print('NaN values present in loss!')
#         print(masked_loss)    

#     loss = torch.sum(masked_loss) / valid_entries

#     return loss



"""
TEST LOSS
"""

if __name__ == '__main__':

    import dataloader
    from processing import *

    MINIKITTI_PATH = '/home/pulp/mini-kitti/'

    trainset    = dataloader.miniKITTI(MINIKITTI_PATH, transform=False, set='train', resolution='360x360', normalize=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

    # Define device, models and training methods
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")

    for i, data in enumerate(trainloader):

        # Get data from dictionary
        imgL    = data['imgL'].type(torch.FloatTensor).to(device)
        imgR    = data['imgR'].type(torch.FloatTensor).to(device)
        depthGT = data['depthGT'].to(device)
        disp    = data['disp'].to(device)
        depth   = data['depth'].to(device)
        fb      = data['fb'].to(device)

        # Disparity to try out the loss
        noise = torch.randn(disp.size()).to(device)
        noisy_disp = disp + noise

        loss, Lap, Lps, La, Lb = MonocularDepthSemiSupervisedLoss(
                    output         = noisy_disp.to(device), 
                    target         = disp.to(device), 
                    imgL           = imgL.to(device), 
                    imgR           = imgR.to(device), 
                    aap            = 0.5, 
                    aps            = 0.5,
                    alpha          = 0.2,
                    invalid_value  = -1,
                    original_width = 360,
                    device         = device)   

        print(f"loss = {loss:.3f}, Lap = {Lap:.3f} (with Lssim = {La:.3f}, Lmod = {Lb:.3f}), Lps = {Lps:.3f}")           

        if (i==5):
            break
