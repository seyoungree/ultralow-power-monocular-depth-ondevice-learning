"""
FUNCTIONS FOR DATA / IMAGE PROCESSING
"""

import torch
import torch.nn.functional as F
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.metrics.regression import *
from ignite.utils import *
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import colorsys
from PIL import Image

"""
UTILS
"""

def bit_count(arr):
     # Make the values type-agnostic (as long as it's integers)
     t = arr.dtype.type
     mask = t(-1)
     s55 = t(0x5555555555555555 & mask)  # Add more digits for 128bit support
     s33 = t(0x3333333333333333 & mask)
     s0F = t(0x0F0F0F0F0F0F0F0F & mask)
     s01 = t(0x0101010101010101 & mask)

     arr = arr - ((arr >> 1) & s55)
     arr = (arr & s33) + ((arr >> 2) & s33)
     arr = (arr + (arr >> 4)) & s0F
     return (arr * s01) >> (8 * (arr.itemsize - 1))

def convert_to_grayscale(src_img):
    img_grayscale = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    return img_grayscale

def crop_center(img,cropx,cropy):
    shape = img.shape
    startx = shape[1]//2-(cropx//2)
    starty = shape[0]//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

# Convert image from float (0,1) to uint8
def convert_img_float_to_uint8(image):
    img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX) 
    img_uint8 = cv2.convertScaleAbs(img_norm)
    return img_uint8

# After reading the LIDAR depth maps of KITTI, this is used to make them dense
def lin_interp(shape, xyd):
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange( n ), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

# REF: https://stackoverflow.com/questions/44382267/how-to-find-the-focal-length-from-camera-matrix
def read_focal_length(file):

    focal_length = 0

    return focal_length


# Clip disparity and depth maps to minimum / maximum meaningful value
def clip_disparity_to_min_value(min_value, disparity_map):

    # Keep invalid values (-1)
    inv_mask = (disparity_map == -1)

    clipped_disparity = disparity_map
    clipped_disparity[clipped_disparity < min_value] = min_value

    # Keep invalid values
    clipped_disparity[inv_mask] = -1

    return clipped_disparity

def clip_depth_to_max_value(max_value, depth_map):

    clipped_depth = depth_map
    clipped_depth[clipped_depth > max_value] = max_value

    return clipped_depth


# Compute depth map from disparity 
def compute_depth_map(cam_par, disparity_map):

    # Find valid values
    mask            = (disparity_map != -1)
    # Compute depth map
    depth_map        = np.ones_like(disparity_map)
    depth_map.fill(-1)
    depth_map[mask] = cam_par / disparity_map[mask]

    return depth_map

# Compute depth map from disparity 
def compute_depth_map_torch(cam_par, disparity_map):

    # Find valid values
    mask            = (disparity_map > 0) #(disparity_map != -1)
    # Compute depth map
    depth_map        = torch.ones_like(disparity_map)
    #depth_map        = depth_map.fill_(-1)
    depth_map[mask]  = cam_par / disparity_map[mask]
    #neg_mask         = torch.logical_not(mask)
    #depth_map[neg_mask] = -1

    return depth_map

# Compute depth maps from predictions in validation
def compute_depth_map_validation(cam_par, disparities, device):
    batch_size, hin, win = disparities.size()
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


# Compute disparity maps from depth predictions in training
def compute_disp_map_training(cam_par, depths, device):
    batch_size, hin, win = depths.size()
    cam_par = cam_par.to(device)

    # Create depth map and fill it with invalid values
    disps = torch.zeros_like(depths).fill_(-1).to(device)
    for smpl in range(batch_size):
        depth = depths[smpl].to(device)
        mask  = (depth > 0)
        mask  = mask.to(device)
        disp  = torch.zeros_like(depth).to(device)
        disp[mask] = cam_par[smpl] / depth[mask]
        disps[smpl] = disp

    return disps


# Count valid points in the ground thruth
def count_gt_valid_points(gt):

    valid_values_mask = (gt > 0)
    valid_points      = np.sum(valid_values_mask)

    return valid_points


# https://stackoverflow.com/questions/7274221/changing-image-hue-with-python-pil
rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = Image.fromarray(image)
    img = img.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')

    return new_img

def warp_right_with_disparity(imgR, disparity, device):
    if imgR.dim() == 3:
        channels, height, width = imgR.size()
    else:
        batch_size, channels, height, width = imgR.size()

    # Remove invalid values from disparity
    disparity = torch.nan_to_num(disparity, nan=0, posinf=0, neginf=0)
    disparity[disparity == -1] = 0
    disparity = disparity.to(device)

    # Define the original axis of the image (on the x)
    x_coord = torch.linspace(0, imgR.shape[3] - 1, imgR.shape[3])
    x_coord = x_coord.to(device)

    # Calculate the distortion matrix
    distortion_matrix = torch.zeros_like(x_coord).to(device)
    distortion_matrix = x_coord + disparity.squeeze(0).squeeze(0)
    distortion_matrix = distortion_matrix.type(torch.int32)

    # Create a mask for valid values
    valid_mask = (distortion_matrix >= 0) & (distortion_matrix < width)
    valid_mask = valid_mask.to(device)
    distortion_matrix *= valid_mask.type(torch.int32)

    # Index into imgR using distortion_matrix
    warped_img = imgR[:, :, torch.arange(height)[:, None], distortion_matrix]
    warped_img.to(device)

    return warped_img

# Scales the disparity according to how the image has been scaled
def warp_right_with_disparity_on_downsampled(imgR, disparity, orig_width, device):
    batch_size, channels, height, width = imgR.size()

    # Remove invalid values from disparity
    disparity = torch.nan_to_num(disparity, nan=0, posinf=0, neginf=0)
    disparity[disparity == -1] = 0

    # Define the original axis of the image (on the x)
    x_coord = torch.linspace(0, imgR.shape[3] - 1, imgR.shape[3]).to(device)

    # Rescale the disparity values according to the downsampling size
    downscale_factor = imgR.size(3) / orig_width
    disparity *= downscale_factor

    # Calculate the distortion matrix
    distortion_matrix = x_coord + disparity.squeeze(0).squeeze(0)
    distortion_matrix = distortion_matrix.type(torch.int32).to(device)

    # Create a mask for valid values
    valid_mask = (distortion_matrix >= 0) & (distortion_matrix < width)
    valid_mask.to(device)
    distortion_matrix *= valid_mask.type(torch.int32)

    # Index into imgR using distortion_matrix
    warped_img = imgR[:, :, torch.arange(height)[:, None], distortion_matrix]
    warped_img.to(device)

    return warped_img


# Compute the SSIM using torch.ignite
# SOURCE: https://pytorch.org/ignite/generated/ignite.metrics.SSIM.html
def compute_ssim_ignite(imgL, imgR):
    
    # Default function
    def eval_step(engine, batch):
        return batch
    
    default_evaluator = Engine(eval_step)
    metric = SSIM(data_range=255)
    metric.attach(default_evaluator, 'ssim')
    state  = default_evaluator.run([[imgL, imgR]])

    return state.metrics['ssim']



"""
TRANSFORMS
"""

# https://stackoverflow.com/questions/38265364/census-transform-in-python-opencv
def censusTransform_3x3 (src_img, ker_h, ker_w, h, w):
    #Convert image to Numpy array
    src_bytes = np.asarray(src_img)

    #Initialize output array
    census = np.zeros((h-2, w-2), dtype='uint8')

    #centre pixels, which are offset by (1, 1)
    cp = src_bytes[1:h-1, 1:w-1]

    #offsets of non-central pixels 
    offsets = [(u, v) for v in range(3) for u in range(3) if not u == 1 == v]

    #Do the pixel comparisons
    for u,v in offsets:
        census = (census << 1) | (src_bytes[v:v+h-2, u:u+w-2] >= cp)

    return census

# https://gist.github.com/charsyam/f42dc0b2e95ba69f0db1ffa021dbaf79
def censusTransform_NxM_slow (src_img, ker_h, ker_w, height, width):
    #Convert image to Numpy array
    hy = ker_h
    wx = ker_w

    image  = np.asarray(src_img)
    census_transf = np.zeros_like(image) # np.zeros([height-hy+1, width-wx])

    for y in range(int(hy/2), height - int(hy/2)):
        for x in range(int(wx/2), width - int(wx/2)):
            census = 0
            shift_count = 0
            #MxN
            for j in range(y - int(hy/2), y + int(hy/2) + 1):
                for i in range(x - int(wx/2), x + int(wx/2) + 1):
                    if shift_count != hy * wx / 2:
                        census <<= 1
                        if image[j][i] < image[y][x]:
                            bit = 1
                        else:
                            bit = 0

                        census = census + bit
                    shift_count += 1
            census_transf[y][x] = census

    return census_transf


def HammingDistance(reference, comparison):

    # Compute the binary XOR between the single pixels in 8 bit
    temp = reference ^ comparison

    # print(f"\nreference: \n{reference}")
    # np.set_printoptions(formatter={'int':bin})
    # print(f"{reference}")
    # np.set_printoptions()

    # print(f"\ncomparison: \n{comparison}")
    # np.set_printoptions(formatter={'int':bin})
    # print(f"{comparison}")
    # np.set_printoptions()

    # print(f"\ntemp: \n{temp}")
    # np.set_printoptions(formatter={'int':bin})
    # print(f"{temp}")
    # np.set_printoptions()

    # For each pixel, count the number of ones
    hamming_distance = bit_count(temp)

    # print(f"\nhamming_distance: \n{hamming_distance}")

    return hamming_distance





# https://www.youtube.com/watch?v=gffZ3S9pBUE
def computeDisparityMap(imgL, imgR):
    
    nDispFactor = 1 # Edit this
    
    #imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    #imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM.create(
        numDisparities = 16*nDispFactor,
        blockSize      = 1 #15
    )
    disparity = stereo.compute(imgL.astype(np.uint8), imgR.astype(np.uint8))

    return disparity


# https://www.youtube.com/watch?v=gffZ3S9pBUE
def SGM_Transform(imgL, imgR):

    window_size = 3
    min_disp    = 2
    nDispFactor = 5    # Very impacting parameter
    num_disp    = 16*nDispFactor - min_disp

    stereo = cv2.StereoSGBM_create(
        minDisparity        = min_disp,             # Minimum considered disparity
        numDisparities      = num_disp,             # Number of disparities
        blockSize           = window_size,          # Size of the block window
        P1                  = 8*3*window_size**2,   # Penalty value 1 (hard to tune)
        P2                  = 32*3*window_size**2,  # Penalty value 2 (hard to tune)
        disp12MaxDiff       = 1,                    # Maximum allowed difference in the left-right disparity check
        preFilterCap        = 63,                   # Truncates disparity values falling outside [-preFilterCap, preFilterCap]
        uniquenessRatio     = 0,                    # % margin by which the best computed cost value should "win" the second best value to conside the found match correct (helps to remove false matches)
        speckleWindowSize   = 0,                    # Max size of smooth disparity rgions to consider their noise speckles and invalidate
        speckleRange        = 2,                    # Maximum disparity variation within specified window
        mode                = cv2.STEREO_SGBM_MODE_HH
    )
    # MODES
    # cv2.STEREO_SGBM_MODE_SGBM: basic full-scale 2-pass dynamic programming algorithm (most accurate but slow)
    # cv2.STEREO_SGBM_MODE_HH: uses Hirschmuller algorithm. Faster than SGBM, with higher memory requirement.
    # cv2.STEREO_SGBM_MODE_SGBM_3WAY: three-way optimized version of SGBM. Faster, less memory, possibly less accurate.
    # cv2.STEREO_SGBM_MODE_HH4: SIMD optimized version of HH. Balances speed and accuracy + HW acceleration.
    # cv2.STEREO_SGBM_MODE_QUARTER: outputs disparity map scaled down by a factor of 4 in H and W. Faster, with lowest resolution.

    disparity = stereo.compute(imgL.astype(np.uint8), imgR.astype(np.uint8)).astype(np.float32) / 16.0

    return disparity


# REF: https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
# REF: https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html
# REF (CODE): https://forum.opencv.org/t/bad-disparity-map-with-sgbm-algorithm/8209/5
def SGM_Transform_Filtered(imgL, imgR):

    window_size = 9
    min_disp    = 8
    nDispFactor = 4    # Very impacting parameter
    num_disp    = 16*nDispFactor - min_disp

    left_matcher = cv2.StereoSGBM_create(
        minDisparity        = min_disp,             # Minimum considered disparity
        numDisparities      = num_disp,             # Number of disparities
        blockSize           = window_size,          # Size of the block window
        P1                  = 8*3*window_size**2,   # Penalty value 1 (hard to tune)
        P2                  = 32*3*window_size**2,  # Penalty value 2 (hard to tune)
        disp12MaxDiff       = 1,                    # Maximum allowed difference in the left-right disparity check
        preFilterCap        = 63,                   # Truncates disparity values falling outside [-preFilterCap, preFilterCap]
        uniquenessRatio     = 0, #15,               # % margin by which the best computed cost value should "win" the second best value to conside the found match correct (helps to remove false matches)
        speckleWindowSize   = 0,                    # Max size of smooth disparity rgions to consider their noise speckles and invalidate
        speckleRange        = 2,                    # Maximum disparity variation within specified window
        mode                = cv2.STEREO_SGBM_MODE_SGBM
    )

    # The weighted least squares (WLS) filter is a well-known edge preserving smoothing technique, but its weights highly depend on the image gradients. 
    # It helps to calculate the smoothing weights for pixels based on both their isotropy and gradients. 

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    alpha = 50
    lmbda = 10
    sigma = 100

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL.astype(np.uint8), imgR.astype(np.uint8)).astype(np.float32) / 16.0
    dispr = right_matcher.compute(imgR.astype(np.uint8), imgL.astype(np.uint8)).astype(np.float32) / 16.0
    #displ = np.int16(displ)
    #dispr = np.int16(dispr)

    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=alpha, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.float32(filteredImg) / 16.0 # np.uint8(filteredImg)

    return filteredImg


# https://github.com/ethz-asl/clubs_dataset_tools/blob/master/python/clubs_dataset_tools/stereo_matching.py
def SGM_Transform_LR_consistency(imgL, imgR):

    window_size = 3
    min_disp    = 2
    nDispFactor = 5    # Very impacting parameter
    num_disp    = 16*nDispFactor - min_disp

    stereo_match = cv2.StereoSGBM_create(
        minDisparity        = min_disp,             # Minimum considered disparity
        numDisparities      = num_disp,             # Number of disparities
        blockSize           = window_size,          # Size of the block window
        P1                  = 8*3*window_size**2,   # Penalty value 1 (hard to tune)
        P2                  = 32*3*window_size**2,  # Penalty value 2 (hard to tune)
        disp12MaxDiff       = 1,                    # Maximum allowed difference in the left-right disparity check
        preFilterCap        = 63,                   # Truncates disparity values falling outside [-preFilterCap, preFilterCap]
        uniquenessRatio     = 0,                    # % margin by which the best computed cost value should "win" the second best value to conside the found match correct (helps to remove false matches)
        speckleWindowSize   = 0,                    # Max size of smooth disparity rgions to consider their noise speckles and invalidate
        speckleRange        = 2,                    # Maximum disparity variation within specified window
        mode                = cv2.STEREO_SGBM_MODE_HH
    )

    right_match = cv2.StereoSGBM_create(
        minDisparity        = -(min_disp+num_disp)+1,   # Minimum considered disparity
        numDisparities      = num_disp,                 # Number of disparities
        blockSize           = window_size,              # Size of the block window
        P1                  = 8*3*window_size**2,       # Penalty value 1 (hard to tune)
        P2                  = 32*3*window_size**2,      # Penalty value 2 (hard to tune)
        disp12MaxDiff       = 1,                        # Maximum allowed difference in the left-right disparity check
        preFilterCap        = 63,                       # Truncates disparity values falling outside [-preFilterCap, preFilterCap]
        uniquenessRatio     = 0,                        # % margin by which the best computed cost value should "win" the second best value to conside the found match correct (helps to remove false matches)
        speckleWindowSize   = 0,                        # Max size of smooth disparity rgions to consider their noise speckles and invalidate
        speckleRange        = 2,                        # Maximum disparity variation within specified window
        mode                = cv2.STEREO_SGBM_MODE_HH
    )

    # "Convolution" effect on the borders (pixels cut)
    border = (window_size-1) + num_disp

    disparity_left  = stereo_match.compute(imgL.astype(np.uint8), imgR.astype(np.uint8)).astype(np.float32) / 16.0
    disparity_right = right_match.compute(imgR.astype(np.uint8), imgL.astype(np.uint8)).astype(np.float32) / 16.0
    disparity_right = np.abs(disparity_right)

    # Set tolerance value for consistency
    eps = 1
    shp = imgL.shape
    # Crop, align and compute mask to enforce consistency
    compare_left  = disparity_left  [:, int(border/2):shp[1]]
    compare_right = disparity_right [:, 0:int(shp[1]-int(border/2))]
    disp_mask     = np.abs(compare_left-compare_right) <= eps 
    # Final disparity map
    disparity = np.ones_like(compare_left) * (-1)
    disparity[disp_mask] = compare_left[disp_mask]
    dshp = disparity.shape
    # To align with the left image, eliminate the right part
    disparity = disparity[:, 0:int(dshp[1]-int(border/2))]

    # figTEST, axTEST = plt.subplots(3, 2)
    # axTEST[0][0].title.set_text("Left-Right Disparity")
    # axTEST[0][1].title.set_text("Left-Right Disparity Cropped")
    # axTEST[1][0].title.set_text("Right-Left Disparity")    
    # axTEST[1][1].title.set_text("Right-Left Disparity Cropped")  
    # axTEST[2][0].title.set_text("Disparity with Left-Right Consistency")
    # axTEST[2][1].title.set_text("SAME AS ITS LEFT")  
    # axTEST[0][0].imshow(disparity_left)
    # axTEST[0][1].imshow(compare_left)
    # axTEST[1][0].imshow(disparity_right)
    # axTEST[1][1].imshow(compare_right)
    # axTEST[2][0].imshow(disparity)
    # axTEST[2][1].imshow(disparity)    

    return disparity


# REF: https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
# REF: https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html
# REF (CODE): https://forum.opencv.org/t/bad-disparity-map-with-sgbm-algorithm/8209/5
# https://github.com/ethz-asl/clubs_dataset_tools/blob/master/python/clubs_dataset_tools/stereo_matching.py
def SGM_Transform_Filtered_LR_consistency(imgL, imgR):

    window_size = 9
    min_disp    = 8
    nDispFactor = 4    # Very impacting parameter
    num_disp    = 16*nDispFactor - min_disp

    left_matcher = cv2.StereoSGBM_create(
        minDisparity        = min_disp,             # Minimum considered disparity
        numDisparities      = num_disp,             # Number of disparities
        blockSize           = window_size,          # Size of the block window
        P1                  = 8*3*window_size**2,   # Penalty value 1 (hard to tune)
        P2                  = 32*3*window_size**2,  # Penalty value 2 (hard to tune)
        disp12MaxDiff       = 1,                    # Maximum allowed difference in the left-right disparity check
        preFilterCap        = 63,                   # Truncates disparity values falling outside [-preFilterCap, preFilterCap]
        uniquenessRatio     = 0, #15,               # % margin by which the best computed cost value should "win" the second best value to conside the found match correct (helps to remove false matches)
        speckleWindowSize   = 0,                    # Max size of smooth disparity rgions to consider their noise speckles and invalidate
        speckleRange        = 2,                    # Maximum disparity variation within specified window
        mode                = cv2.STEREO_SGBM_MODE_SGBM
    )

    right_matcher = cv2.StereoSGBM_create(
        minDisparity        = -(min_disp+num_disp)+1,   # Minimum considered disparity
        numDisparities      = num_disp,                 # Number of disparities
        blockSize           = window_size,              # Size of the block window
        P1                  = 8*3*window_size**2,       # Penalty value 1 (hard to tune)
        P2                  = 32*3*window_size**2,      # Penalty value 2 (hard to tune)
        disp12MaxDiff       = 1,                        # Maximum allowed difference in the left-right disparity check
        preFilterCap        = 63,                       # Truncates disparity values falling outside [-preFilterCap, preFilterCap]
        uniquenessRatio     = 0,                        # % margin by which the best computed cost value should "win" the second best value to conside the found match correct (helps to remove false matches)
        speckleWindowSize   = 0,                        # Max size of smooth disparity rgions to consider their noise speckles and invalidate
        speckleRange        = 2,                        # Maximum disparity variation within specified window
        mode                = cv2.STEREO_SGBM_MODE_HH
    )

    # The weighted least squares (WLS) filter is a well-known edge preserving smoothing technique, but its weights highly depend on the image gradients. 
    # It helps to calculate the smoothing weights for pixels based on both their isotropy and gradients. 

    # FILTER Parameters
    alpha = 40
    lmbda = 10
    sigma = 100
    disc_radius = 1
    lrc_thresh  = 1

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    wls_filter.setDepthDiscontinuityRadius(disc_radius)
    wls_filter.setLRCthresh(lrc_thresh)

    displ = left_matcher.compute(imgL.astype(np.uint8), imgR.astype(np.uint8)).astype(np.float32) / 16.0
    dispr = right_matcher.compute(imgR.astype(np.uint8), imgL.astype(np.uint8)).astype(np.float32) / 16.0
    #displ = np.int16(displ)
    #dispr = np.int16(dispr)

    filteredImg = wls_filter.filter(displ, imgL, None, dispr, None, imgR)  # important to put "imgL" here!!!
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=alpha, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.float32(filteredImg) # np.uint8(filteredImg)

    return filteredImg



"""
TO BE DEBUGGED
"""

# TODO: Create / Debug NxM Optimized Census Transform 

# https://stackoverflow.com/questions/38265364/census-transform-in-python-opencv
def censusTransform_NxM (src_img, ker_h, ker_w, h, w):
    #Convert image to Numpy array
    src_bytes = np.asarray(src_img)
    kh = ker_h
    kw = ker_w

    #Initialize output array
    census = np.zeros((h-(kh-1), w-(kw-1)), dtype='uint8')

    #centre pixels, which are offset by (1, 1)
    cp = src_bytes[1:h-1, 1:w-1]

    #offsets of non-central pixels 
    offsets = [(u, v) for v in range(kh) for u in range(kw) if not u == 1 == v]

    #Do the pixel comparisons
    for u,v in offsets:
        census = (census << 1) | (src_bytes[v:v+h-(kh-1), u:u+w-(kw-1)] >= cp)

    return census

def census_transform_optimized(img):
    """
    Calculates the 9x7 census transform of an image using matrix operations.

    Args:
        img (numpy.ndarray): Input grayscale image.

    Returns:
        numpy.ndarray: Census transformed image.
    """
    height, width = img.shape

    print(f"{height}, {width}")

    # Create a sliding window view of the image
    window_shape = (9, 7)
    window_view = np.lib.stride_tricks.sliding_window_view(img, window_shape)
    
    #print(f"window_view.shape = {window_view.shape}")
    #print(f"{window_view}")

    # Calculate the binary codes for each window
    center_value = window_view[:, :, 4, 3]
    
    # print(f"\n{center_value}, {center_value.shape}")

    import pdb; pdb.set_trace()
    
    # print(f"{neighbor_values}, {neighbor_values.shape}")

    binary_codes = (window_view <= center_value)

    #print(f"binary_codes.shape = {binary_codes.shape}")

    # Convert binary codes to integers
    powers_of_two = 2 ** np.arange(63, -1, -1, dtype=np.uint64)
    census_img = np.sum(binary_codes * powers_of_two, axis=(2, 3))

    return census_img




"""
TEST FUNCTION
"""
if __name__ == '__main__':

    ham_ref  = np.array([[100, 123], [68, 92]])
    ham_comp = np.array([[136, 104], [63, 87]])

    hamdist  = HammingDistance(ham_ref, ham_comp)

    print(f"\nFinal Hamming Distance = {hamdist}")

