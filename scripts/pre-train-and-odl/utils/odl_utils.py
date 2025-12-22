import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def downsample_disparity(disp_maps, size=[8, 8], mode=transforms.InterpolationMode.NEAREST):
    """
    Function used to downsample the proxy disparity maps to the size of a simulated low-resolution sensor
    """
    low_res_disp_maps = transforms.functional.resize(img=disp_maps, size=[size[0], size[1]], interpolation=mode)
    return low_res_disp_maps


def upsample_tof_data(tof_disp_map, size=[48, 48], mode='nearest'):
    """
    Function to upsample the data of a low-resolution sensor to the size of the output of the uPyD-Net model
    """
    if mode == 'nearest':
        disp_map = transforms.functional.resize(img=tof_disp_map, size=[size[0], size[1]], interpolation=transforms.InterpolationMode.NEAREST)
    elif mode == 'bilinear':
        disp_map = transforms.functional.resize(img=tof_disp_map, size=[size[0], size[1]], interpolation=transforms.InterpolationMode.BILINEAR)
    elif mode == 'sparse':
        # TODO: complete the method to map only the valid value to a specific pixel with respect to the ToF sensor of STM
        # https://www.st.com/en/imaging-and-photonics-solutions/vl53l5cx.html
        print(f"[upsample_tof_data] 'sparse' method incomplete!!")
        exit()
        disp_map = torch.ones(tof_disp_map[0], size[0], size[1]) * (-1)
        sensor_size = tof_disp_map.size()
    else:
        print("[upsample_tof_data] Invalid upsample option!!")
        exit()

    return disp_map

# Transforms the ground truth in a similar shape of the ToF sensor: takes the minimum in each block
def simulate_tof_sensor_ground_truth(ground_truth, sensor_resolution_h, sensor_resolution_w, output_shape_h, output_shape_w, invalid_value):

    #import pdb; pdb.set_trace()

    with torch.no_grad():
        # Define the ratio between the input and the sensor resolution
        size_in = ground_truth.size()
        h_ratio = int(size_in[-2] / sensor_resolution_h)
        w_ratio = int(size_in[-1] / sensor_resolution_w)
        # Pre-process distances to mask invalid values
        mask = (ground_truth == invalid_value)
        filt_gt = ground_truth
        filt_gt[mask] = 200.0
        # Find minimum value for each block
        min_search = torch.nn.MaxPool2d(
                        kernel_size = (h_ratio, w_ratio),
                        stride      = (h_ratio, w_ratio),
                        padding     = 0,
                        dilation    = 1,
                    )
        tof_simulated_data = min_search(- filt_gt)
        tof_simulated_data = -tof_simulated_data
        remask = (tof_simulated_data == 200.0)
        tof_simulated_data[remask] = invalid_value
        # Reshape data to output shape
        tof_simulated_data = transforms.functional.resize(img=tof_simulated_data, size=[output_shape_h, output_shape_w], interpolation=transforms.InterpolationMode.NEAREST)
        
        #fig, ax = plt.subplots(2,1)
        #show_gt = ground_truth.squeeze(0)
        #show_tof = tof_simulated_data.squeeze(0)
        #ax[0].imshow(show_gt)
        #ax[1].imshow(show_tof)
        #plt.show()

    return tof_simulated_data

# Transforms the ground truth in a similar shape of the ToF sensor: takes the minimum in each block
def simulate_tof_sensor_disparity(ground_truth, sensor_resolution_h, sensor_resolution_w, output_shape_h, output_shape_w, invalid_value, upsample_strategy):

    #import pdb; pdb.set_trace()

    with torch.no_grad():
        # Define the ratio between the input and the sensor resolution
        size_in = ground_truth.size()
        h_ratio = int(size_in[-2] / sensor_resolution_h)
        w_ratio = int(size_in[-1] / sensor_resolution_w)
        # Pre-process distances to mask invalid values
        mask = (ground_truth == invalid_value)
        filt_gt = ground_truth
        #filt_gt[mask] = -2
        # Find minimum value for each block
        max_search = torch.nn.MaxPool2d(
                        kernel_size = (h_ratio, w_ratio),
                        stride      = (h_ratio, w_ratio),
                        padding     = 0,
                        dilation    = 1,
                    )
        tof_simulated_data = max_search(filt_gt)
        #remask = (tof_simulated_data == -2)
        #tof_simulated_data[remask] = invalid_value
        # Reshape data to output shape
        if upsample_strategy == 'nearest':
            tof_simulated_data = transforms.functional.resize(img=tof_simulated_data, size=[output_shape_h, output_shape_w], interpolation=transforms.InterpolationMode.NEAREST)
        elif upsample_strategy == 'bilinear':
            tof_simulated_data = transforms.functional.resize(img=tof_simulated_data, size=[output_shape_h, output_shape_w], interpolation=transforms.InterpolationMode.BILINEAR)
        elif upsample_strategy == 'sparse':
            # TODO: complete the method to map only the valid value to a specific pixel with respect to the ToF sensor of STM
            # https://www.st.com/en/imaging-and-photonics-solutions/vl53l5cx.html
            print(f"[upsample_tof_data] 'sparse' method incomplete!!")
            exit()
            tof_simulated_data = torch.ones(tof_disp_map[0], size[0], size[1]) * (-1)
            sensor_size = tof_disp_map.size()
        else:
            print("[upsample_tof_data] Invalid upsample option!!")
            exit()
        
        #fig, ax = plt.subplots(2,1)
        #show_gt = ground_truth[0].squeeze(0)
        #show_tof = tof_simulated_data[0].squeeze(0)
        #show_gt = show_gt.cpu().float()
        #show_tof = show_tof.cpu().float()
        #ax[0].imshow(show_gt)
        #ax[1].imshow(show_tof)
        #plt.show()

    return tof_simulated_data


"""
FUNCTIONS FOR MULTI-TARGET TOF MEASUREMENTS
"""

# Compute depth maps from predictions in validation
def compute_depth_map_validation_tof(cam_par, disparities, device):
    batch_size, num_max, hin, win = disparities.size()
    cam_par = cam_par.to(device)

    # Create depth map and fill it with invalid values
    depths = torch.zeros_like(disparities).fill_(-1).to(device)
    for smpl in range(batch_size):
        disp = disparities[smpl].to(device)
        mask = (disp > 0)
        mask = mask.to(device)
        dpth = torch.zeros_like(disp).to(device)
        dpth[mask] = cam_par[smpl] / disp[mask]
        depths[smpl] = dpth

    return depths

def convert_depth_label_to_multi_target_disparity(depth_label, fb, num_minima, device, dtype):

    sensor_tolerance = 0.6   # 600 mm between one minimum and the following one: https://www.st.com/resource/en/user_manual/um2884-a-guide-to-using-the-vl53l5cx-multizone-timeofflight-ranging-sensor-with-a-wide-field-of-view-ultra-lite-driver-uld-stmicroelectronics.pdf

    if depth_label.dim() == 2:
        disp = torch.zeros(num_minima, 8, 8).to(device)
        if dtype == 'fp16':
            disp = disp.half()
        elif dtype == 'bfloat16':
            disp = disp.bfloat16()
    elif depth_label.dim() == 3:
        disp = torch.zeros(depth_label.size()[0], num_minima, 8, 8).to(device)
        if dtype == 'fp16':
            disp = disp.half()
        elif dtype == 'bfloat16':
            disp = disp.bfloat16()
    
    temp_depth = torch.zeros_like(disp).to(device)
    if dtype == 'fp16':
        temp_depth = temp_depth.half()
    elif dtype == 'bfloat16':
        temp_depth = temp_depth.bfloat16()

    # Define pooling to compute minpool
    pool = torch.nn.MaxPool2d(kernel_size=6, stride=6).to(device)
    if dtype == 'fp16':
        pool = pool.half()
    elif dtype == 'bfloat16':
        pool = pool.bfloat16()

    # Substitute the invalid depths with 80 m before minpool
    inv_vals = (depth_label == -1)
    depth_label[inv_vals] = 80.0

    # # Find minima
    # for minimum in range(num_minima):
    #     temp_depth[:, minimum, :, :] = (-pool(-depth_label))
    #     # Mask the label in the range of the first minimum
    #     for elem in range(depth_label.size()[0]):
    #         for y_sec in range(8):
    #             for x_sec in range(8):
    #                 for y_loc in range(6):
    #                     for x_loc in range(6):
    #                         if depth_label[elem, (6*y_sec+y_loc), (6*x_sec+x_loc)] < (temp_depth[elem, minimum, y_sec, x_sec] + sensor_tolerance):
    #                             depth_label[elem, (6*y_sec+y_loc), (6*x_sec+x_loc)] = 80.0     # Set to maximum value

    # Find minima
    for minimum in range(num_minima):
        temp_depth[:, minimum, :, :] = (-pool(-depth_label))
        # Reshape depth_label and temp_depth for vectorized operations
        depth_label_reshaped = depth_label.view(depth_label.size(0), 8, 6, 8, 6)
        temp_depth_expanded = temp_depth[:, minimum, :, :].unsqueeze(2).unsqueeze(4)
        # Create a mask where depth_label is less than temp_depth + sensor_tolerance
        mask = depth_label_reshaped < (temp_depth_expanded + sensor_tolerance)
        # Set depth_label to 80.0 where the mask is True
        depth_label_reshaped[mask] = 80.0
        # Reshape depth_label back to its original size
        depth_label = depth_label_reshaped.view_as(depth_label)

    # # Transform temp_depth into a disparity map
    # for elem in range(depth_label.size()[0]):
    #     disp[elem, :, :, :] = fb[elem, :, :] / temp_depth[elem, :, :, :]

    # Track where invalid values are
    inv_vals_post = (temp_depth == 80.0)

    # Transform temp_depth into a disparity map using element-wise division
    disp = fb.unsqueeze(2) / temp_depth

    # Set invalid values to -1
    disp[inv_vals_post] = -1

    return disp


def convert_depth_label_to_multi_target_disparity_indoor(depth_label_in, fb, num_minima, device, dtype):

    sensor_tolerance = 0.6   # 600 mm between one minimum and the following one: https://www.st.com/resource/en/user_manual/um2884-a-guide-to-using-the-vl53l5cx-multizone-timeofflight-ranging-sensor-with-a-wide-field-of-view-ultra-lite-driver-uld-stmicroelectronics.pdf

    depth_label = torch.tensor(depth_label_in).to(device).clone()

    if depth_label.dim() == 2:
        disp = torch.zeros(num_minima, 8, 8).to(device)
        if dtype == 'fp16':
            disp = disp.half()
        elif dtype == 'bfloat16':
            disp = disp.bfloat16()
    elif depth_label.dim() == 3:
        disp = torch.zeros(depth_label.size()[0], num_minima, 8, 8).to(device)
        if dtype == 'fp16':
            disp = disp.half()
        elif dtype == 'bfloat16':
            disp = disp.bfloat16()
    
    temp_depth = torch.zeros_like(disp).to(device)
    if dtype == 'fp16':
        temp_depth = temp_depth.half()
    elif dtype == 'bfloat16':
        temp_depth = temp_depth.bfloat16()

    # Define pooling to compute minpool
    pool = torch.nn.MaxPool2d(kernel_size=6, stride=6).to(device)
    if dtype == 'fp16':
        pool = pool.half()
    elif dtype == 'bfloat16':
        pool = pool.bfloat16()

    # Substitute the invalid depths with 80 m before minpool
    inv_vals = (depth_label == -1)
    depth_label[inv_vals] = 5.0

    # # Find minima
    # for minimum in range(num_minima):
    #     temp_depth[:, minimum, :, :] = (-pool(-depth_label))
    #     # Mask the label in the range of the first minimum
    #     for elem in range(depth_label.size()[0]):
    #         for y_sec in range(8):
    #             for x_sec in range(8):
    #                 for y_loc in range(6):
    #                     for x_loc in range(6):
    #                         if depth_label[elem, (6*y_sec+y_loc), (6*x_sec+x_loc)] < (temp_depth[elem, minimum, y_sec, x_sec] + sensor_tolerance):
    #                             depth_label[elem, (6*y_sec+y_loc), (6*x_sec+x_loc)] = 80.0     # Set to maximum value

    # Find minima
    for minimum in range(num_minima):
        temp_depth[:, minimum, :, :] = (-pool(-depth_label))
        # Reshape depth_label and temp_depth for vectorized operations
        depth_label_reshaped = depth_label.view(depth_label.size(0), 8, 6, 8, 6)
        temp_depth_expanded = temp_depth[:, minimum, :, :].unsqueeze(2).unsqueeze(4)
        # Create a mask where depth_label is less than temp_depth + sensor_tolerance
        mask = depth_label_reshaped < (temp_depth_expanded + sensor_tolerance)
        # Set depth_label to 80.0 where the mask is True
        depth_label_reshaped[mask] = 5.0
        # Reshape depth_label back to its original size
        depth_label = depth_label_reshaped.view_as(depth_label)
        
        # DEBUG
        print(f"Minimum = {minimum}")
        print(torch.sum(mask) / mask.numel())
        figg, axx = plt.subplots(1, 5)
        axx[0].imshow(temp_depth[0][0].float().cpu().numpy())
        axx[1].imshow(temp_depth[0][1].float().cpu().numpy())
        axx[2].imshow(temp_depth[0][2].float().cpu().numpy())
        axx[3].imshow(temp_depth[0][3].float().cpu().numpy())
        axx[4].imshow(depth_label[0].float().cpu().numpy())
        plt.show()

    # # Transform temp_depth into a disparity map
    # for elem in range(depth_label.size()[0]):
    #     disp[elem, :, :, :] = fb[elem, :, :] / temp_depth[elem, :, :, :]

    # Track where invalid values are
    inv_vals_post = (temp_depth == 5.0)

    # Transform temp_depth into a disparity map using element-wise division
    disp = fb.unsqueeze(2) / temp_depth

    # Set invalid values to min disparity
    MIN_DISP = (fb / 5.0).squeeze(1).squeeze(1)
    disp[inv_vals_post] = MIN_DISP[0]

    return disp



def convert_depth_label_to_multi_target_depth(depth_label, num_minima, device, dtype):

    sensor_tolerance = 0.6   # 600 mm between one minimum and the following one: https://www.st.com/resource/en/user_manual/um2884-a-guide-to-using-the-vl53l5cx-multizone-timeofflight-ranging-sensor-with-a-wide-field-of-view-ultra-lite-driver-uld-stmicroelectronics.pdf

    if depth_label.dim() == 2:
        disp = torch.zeros(num_minima, 8, 8).to(device)
        if dtype == 'fp16':
            disp = disp.half()
        elif dtype == 'bfloat16':
            disp = disp.bfloat16()
    elif depth_label.dim() == 3:
        disp = torch.zeros(depth_label.size()[0], num_minima, 8, 8).to(device)
        if dtype == 'fp16':
            disp = disp.half()
        elif dtype == 'bfloat16':
            disp = disp.bfloat16()
    
    temp_depth = torch.zeros_like(disp).to(device)
    if dtype == 'fp16':
        temp_depth = temp_depth.half()
    elif dtype == 'bfloat16':
        temp_depth = temp_depth.bfloat16()

    # Define pooling to compute minpool
    pool = torch.nn.MaxPool2d(kernel_size=6, stride=6).to(device)
    if dtype == 'fp16':
        pool = pool.half()
    elif dtype == 'bfloat16':
        pool = pool.bfloat16()

    # # Check if the depth_label is the ground truth, in that case reduce its size
    # if depth_label.size()[-2] > 48 and depth_label.size()[-1] > 48:
    #     depth_label = transforms.functional.resize(img=depth_label, size=[48, 48], interpolation=transforms.InterpolationMode.NEAREST)

    # Substitute the invalid depths with 80 m before minpool
    inv_vals = (depth_label == -1)
    inv_vals = inv_vals.to(device)
    depth_label[inv_vals] = 80.0

    # Find minima
    for minimum in range(num_minima):
        temp_depth[:, minimum, :, :] = (-pool(-depth_label))
        # Reshape depth_label and temp_depth for vectorized operations
        depth_label_reshaped = depth_label.view(depth_label.size(0), 8, 6, 8, 6)
        temp_depth_expanded = temp_depth[:, minimum, :, :].unsqueeze(2).unsqueeze(4)
        # Create a mask where depth_label is less than temp_depth + sensor_tolerance
        mask = depth_label_reshaped < (temp_depth_expanded + sensor_tolerance)
        # Set depth_label to 80.0 where the mask is True
        depth_label_reshaped[mask] = 80.0
        # Reshape depth_label back to its original size
        depth_label = depth_label_reshaped.view_as(depth_label)

    # # Transform temp_depth into a disparity map
    # for elem in range(depth_label.size()[0]):
    #     disp[elem, :, :, :] = fb[elem, :, :] / temp_depth[elem, :, :, :]

    return temp_depth

# Multi-target ToF sensor simulation for KITTI's ground truth data
def convert_depth_label_to_multi_target_depth_GT(depth_label, num_minima, device, dtype):

    sensor_tolerance = 0.6   # 600 mm between one minimum and the following one: https://www.st.com/resource/en/user_manual/um2884-a-guide-to-using-the-vl53l5cx-multizone-timeofflight-ranging-sensor-with-a-wide-field-of-view-ultra-lite-driver-uld-stmicroelectronics.pdf

    if depth_label.dim() == 2:
        disp = torch.zeros(num_minima, 8, 8).to(device)
        if dtype == 'fp16':
            disp = disp.half()
        elif dtype == 'bfloat16':
            disp = disp.bfloat16()
    elif depth_label.dim() == 3:
        disp = torch.zeros(depth_label.size()[0], num_minima, 8, 8).to(device)
        if dtype == 'fp16':
            disp = disp.half()
        elif dtype == 'bfloat16':
            disp = disp.bfloat16()
    
    temp_depth = torch.zeros_like(disp).to(device)
    if dtype == 'fp16':
        temp_depth = temp_depth.half()
    elif dtype == 'bfloat16':
        temp_depth = temp_depth.bfloat16()

    pool_ker_size = int(depth_label.size()[-1] / 8)
    pool_stride   = int(pool_ker_size)

    # Define pooling to compute minpool
    pool = torch.nn.MaxPool2d(kernel_size=pool_ker_size, stride=pool_stride).to(device)
    if dtype == 'fp16':
        pool = pool.half()
    elif dtype == 'bfloat16':
        pool = pool.bfloat16()

    # Substitute the invalid depths with 80 m before minpool (set them to 100.0 to recognise them)
    inv_vals = (depth_label == 0.0)
    inv_vals = inv_vals.to(device)
    depth_label[inv_vals] = 100.0

    # Find minima
    for minimum in range(num_minima):
        temp_depth[:, minimum, :, :] = (-pool(-depth_label))
        # Reshape depth_label and temp_depth for vectorized operations
        depth_label_reshaped = depth_label.view(depth_label.size(0), 8, pool_ker_size, 8, pool_ker_size)
        temp_depth_expanded = temp_depth[:, minimum, :, :].unsqueeze(2).unsqueeze(4)
        # Create a mask where depth_label is less than temp_depth + sensor_tolerance
        mask = depth_label_reshaped < (temp_depth_expanded + sensor_tolerance)
        # Set depth_label to 100.0 where the mask is True
        depth_label_reshaped[mask] = 100.0
        # Reshape depth_label back to its original size
        depth_label = depth_label_reshaped.view_as(depth_label)

    # Reset invalid values (100.0) to 0 
    invalid_mask = (temp_depth == 100.0)
    temp_depth[invalid_mask] = 0.0

    return temp_depth



# # Module to transform the output of the model to something compatible with the Nx8x8 labels
# class PredictionMultitargetFinder(torch.nn.Module):

#     def __init__(self, device, datatype):
#         super(PredictionMultitargetFinder, self).__init__()
#         self.pool     = torch.nn.AvgPool2d(kernel_size=6, stride=6)
#         self.device   = device
#         self.datatype = datatype

#     def forward(self, pred, label):
#         if label.dim() == 3:
#             batch_size = 1
#             num_maxes = label.size()[0]
#         elif label.dim() == 4:
#             batch_size = label.size()[0]
#             num_maxes  = label.size()[1]
 
#         temp_disp = torch.zeros(batch_size, num_maxes, pred.size()[-2], pred.size()[-1]).to(self.device)
#         out_data  = torch.zeros_like(label).to(self.device)
#         if self.datatype == 'fp16':
#             temp_disp = temp_disp.half()
#             out_data  = out_data.half()
#         elif self.datatype == 'bfloat16':
#             temp_disp = temp_disp.bfloat16()
#             out_data  = out_data.bfloat16()

#         #import pdb; pdb.set_trace()
        
#         # Mask prediction to keep values greater than each range 
#         # in each iteration (disparity)
#         for maxm in range(num_maxes):
#             for elem in range(batch_size):
#                 for y_sec in range(8):
#                     for x_sec in range(8):
#                         for y_loc in range(6):
#                             for x_loc in range(6):
#                                 if maxm == 0:
#                                     if (pred[elem, (6*y_sec+y_loc), (6*x_sec+x_loc)] >= label[elem, maxm, y_sec, x_sec]):
#                                         temp_disp[elem, maxm, (6*y_sec+y_loc), (6*x_sec+x_loc)] = pred[elem, (6*y_sec+y_loc), (6*x_sec+x_loc)]
#                                 elif maxm < (num_maxes-1):
#                                     if (pred[elem, (6*y_sec+y_loc), (6*x_sec+x_loc)] >= label[elem, maxm, y_sec, x_sec]) and \
#                                        (pred[elem, (6*y_sec+y_loc), (6*x_sec+x_loc)] < label[elem, maxm-1, y_sec, x_sec]):
#                                         temp_disp[elem, maxm, (6*y_sec+y_loc), (6*x_sec+x_loc)] = pred[elem, (6*y_sec+y_loc), (6*x_sec+x_loc)]
#                                 elif maxm == (num_maxes-1): 
#                                     if (pred[elem, (6*y_sec+y_loc), (6*x_sec+x_loc)] < label[elem, maxm, y_sec, x_sec]):
#                                         temp_disp[elem, maxm, (6*y_sec+y_loc), (6*x_sec+x_loc)] = pred[elem, (6*y_sec+y_loc), (6*x_sec+x_loc)]

#         #import pdb; pdb.set_trace()

#         # Apply pooling to reduce the size to the one of the label
#         out_data = self.pool(temp_disp)

#         return out_data



# Module to transform the output of the model to something compatible with the Nx8x8 labels
class PredictionMultitargetFinder(torch.nn.Module):

    def __init__(self, device, datatype):
        super(PredictionMultitargetFinder, self).__init__()
        self.pool     = torch.nn.MaxPool2d(kernel_size=6, stride=6)
        self.device   = device
        self.datatype = datatype

    def forward(self, pred, label):
        if label.dim() == 3:
            batch_size = 1
            num_maxes = label.size()[0]
        elif label.dim() == 4:
            batch_size = label.size()[0]
            num_maxes  = label.size()[1]
 
        temp_disp = torch.zeros(batch_size, num_maxes, pred.size()[-2], pred.size()[-1]).to(self.device)
        out_data  = torch.zeros_like(label).to(self.device)
        if self.datatype == 'fp16':
            temp_disp = temp_disp.half()
            out_data  = out_data.half()
        elif self.datatype == 'bfloat16':
            temp_disp = temp_disp.bfloat16()
            out_data  = out_data.bfloat16()

        label_expanded = torch.nn.functional.interpolate(label, size=(48, 48), mode='nearest')
        
        # Mask prediction to keep values greater than each range 
        # in each iteration (disparity)
        if num_maxes == 1:
            temp_disp = pred.unsqueeze(1)
        else:
            for maxm in range(num_maxes):
                for elem in range(batch_size):
                    if maxm == 0:
                        mask = (pred[elem, :, :] >= label_expanded[elem, maxm, :, :])
                        temp_disp[elem, maxm, mask] = pred[elem, mask]
                    elif maxm < num_maxes-1:
                        mask = torch.logical_and(
                            pred[elem, :, :] >= label_expanded[elem, maxm, :, :], 
                            pred[elem, :, :] < label_expanded[elem, maxm-1, :, :])
                        temp_disp[elem, maxm, mask] = pred[elem, mask]
                    else:
                        mask = (pred[elem, :, :] < label_expanded[elem, maxm, :, :])
                        temp_disp[elem, maxm, mask] = pred[elem, mask]

        # Apply pooling to reduce the size to the one of the label
        out_data = self.pool(temp_disp)

        import pdb; pdb.set_trace()

        return out_data


# class PredictionMultitargetFinder(torch.nn.Module):

#     def __init__(self, device, datatype):
#         super(PredictionMultitargetFinder, self).__init__()
#         self.pool     = torch.nn.MaxPool2d(kernel_size=6, stride=6)
#         self.device   = device
#         self.datatype = datatype

#     def forward(self, pred, label):
#         if label.dim() == 3:
#             batch_size = 1
#             num_maxes = label.size()[0]
#         elif label.dim() == 4:
#             batch_size = label.size()[0]
#             num_maxes  = label.size()[1]
 
#         temp_disp = torch.zeros(batch_size, num_maxes, pred.size()[-2], pred.size()[-1]).to(self.device)
#         out_data  = torch.zeros_like(label).to(self.device)
#         if self.datatype == 'fp16':
#             temp_disp = temp_disp.half()
#             out_data  = out_data.half()
#         elif self.datatype == 'bfloat16':
#             temp_disp = temp_disp.bfloat16()
#             out_data  = out_data.bfloat16()

#         temp_disp_rotate=temp_disp.permute(1,0,2,3)
#         new_label=label.repeat(1,1,6,6)  
#         selected_for_maxm=[]
#         for maxm in range(num_maxes):
#             if maxm == 0:
#                 selected_for_maxm.append(pred >= new_label[:, maxm]) 
#             elif maxm < (num_maxes-1):
#                 selected_for_maxm.append(torch.logical_and(pred <= new_label[:, maxm],pred > new_label[:, maxm - 1])) 
#             elif maxm == (num_maxes-1): 
#                 selected_for_maxm.append(pred <= new_label[:, maxm]) 

#         for maxm_idx,selected in enumerate(selected_for_maxm):
#             temp_disp_rotate[maxm_idx,selected]=pred[selected]
        
#         temp_disp=temp_disp_rotate.permute(1,0,2,3)

#         import pdb; pdb.set_trace()

#         # Apply pooling to reduce the size to the one of the label
#         out_data = self.pool(temp_disp)

#         return out_data














"""
TESTING AND VERIFICATION
"""

VERIFY_TOF_SIMULATION_FUNCTION = True

if __name__ == '__main__':

    """
    VERIFY simulate_tof_sensor_disparity()
    """
    if VERIFY_TOF_SIMULATION_FUNCTION:

        DISPARITY_RESOLUTION = 16
        SENSOR_RESOLUTION = 4

        fake_disparity_map_easy = torch.ones(DISPARITY_RESOLUTION, DISPARITY_RESOLUTION)
        fake_disparity_map = torch.randint(low=0, high=10, size=[SENSOR_RESOLUTION, SENSOR_RESOLUTION])
        fake_disparity_map = transforms.functional.resize(img=fake_disparity_map.unsqueeze(0), size=[DISPARITY_RESOLUTION, DISPARITY_RESOLUTION], interpolation=transforms.InterpolationMode.BILINEAR)
        fake_disparity_map = fake_disparity_map.squeeze(0)

        invalid_mask = torch.zeros(DISPARITY_RESOLUTION, DISPARITY_RESOLUTION, dtype=torch.int)
        # Define invalid areas
        # invalid_mask[0:7, 1:8] = 1
        # invalid_mask[12:14, 12:16] = 1
        # invalid_mask[21:26, 3:8]  = 1
        # invalid_mask[29:34, 16:32] = 1
        invalid_mask[0:6, 0:5] = 1
        invalid_mask[DISPARITY_RESOLUTION-7:DISPARITY_RESOLUTION, DISPARITY_RESOLUTION-7:DISPARITY_RESOLUTION] = 1

        # Set print options
        torch.set_printoptions(threshold=torch.numel(fake_disparity_map))

        """
        PRINT INVALID MASK
        """
        print("\nINVALID MASK")
        for row in invalid_mask:
            print(row.tolist())  # Convert row to list for a cleaner output

        # Further fill with invalid values
        fake_disparity_map_easy[invalid_mask == 1] = -1
        fake_disparity_map[invalid_mask == 1] = -1

        """
        TEST 'EASY' CASE
        """
        print("\n(A) FAKE DISPARITY MAP (EASY)")
        for row in fake_disparity_map_easy:
            print(row.tolist())  # Convert row to list for a cleaner output

        print("\n\n(A) SIMULATED TOF DISPARITY MAP (EASY)")
        fake_disparity_map_easy = fake_disparity_map_easy.unsqueeze(0)
        tof_disp_map_easy = simulate_tof_sensor_disparity(fake_disparity_map_easy, SENSOR_RESOLUTION, SENSOR_RESOLUTION, DISPARITY_RESOLUTION, DISPARITY_RESOLUTION, -1, 'nearest')
        tof_disp_map_easy = tof_disp_map_easy.squeeze(0)
        for row in tof_disp_map_easy:
            print(row.tolist())  # Convert row to list for a cleaner output

        # Count the number of disparities that remain the same
        num_equal_pixels_easy = torch.sum(torch.eq(fake_disparity_map_easy, tof_disp_map_easy))
        tot_pixels_easy = torch.numel(fake_disparity_map_easy)
        perc_equal_pixels_easy = num_equal_pixels_easy / tot_pixels_easy
        print(f"\nTOTAL INVALID PIXELS BEFORE SIMULATION: {torch.sum(fake_disparity_map_easy == -1)} ({torch.sum(fake_disparity_map_easy == -1) / tot_pixels_easy:.2f})%")
        print(f"TOTAL INVALID PIXELS AFTER SIMULATION: {torch.sum(tof_disp_map_easy == -1)} ({torch.sum(tof_disp_map_easy == -1) / tot_pixels_easy:.2f})%")
        print(f"EQUAL ELEMENTS AFTER TOF SIMULATION: {num_equal_pixels_easy} / {tot_pixels_easy} ({perc_equal_pixels_easy:.2f}%)")

        """
        TEST 'COMPLEX' CASE
        """
        print("\n\n\n(B) FAKE DISPARITY MAP (COMPLEX)")
        for row in fake_disparity_map:
            print(row.tolist())  # Convert row to list for a cleaner output

        print("\n(B) SIMULATED TOF DISPARITY MAP (COMPLEX)")
        fake_disparity_map = fake_disparity_map.unsqueeze(0)
        tof_disp_map = simulate_tof_sensor_disparity(fake_disparity_map, SENSOR_RESOLUTION, SENSOR_RESOLUTION, DISPARITY_RESOLUTION, DISPARITY_RESOLUTION, -1, 'nearest')
        tof_disp_map = tof_disp_map.squeeze(0)
        for row in tof_disp_map:
            print(row.tolist())  # Convert row to list for a cleaner output

        # Count the number of disparities that remain the same
        num_equal_pixels = torch.sum(torch.eq(fake_disparity_map, tof_disp_map))
        tot_pixels = torch.numel(fake_disparity_map)
        perc_equal_pixels = num_equal_pixels / tot_pixels
        print(f"\nTOTAL INVALID PIXELS BEFORE SIMULATION: {torch.sum(fake_disparity_map == -1)} ({torch.sum(fake_disparity_map == -1) / tot_pixels:.2f})%")
        print(f"TOTAL INVALID PIXELS AFTER SIMULATION: {torch.sum(tof_disp_map == -1)} ({torch.sum(tof_disp_map == -1) / tot_pixels:.2f})%")
        print(f"EQUAL ELEMENTS AFTER TOF SIMULATION: {num_equal_pixels} / {tot_pixels} ({perc_equal_pixels:.2f}%)")
