import math

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from ex1_utils import gausssmooth


def generate_responses_1():
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[70, 50] = 1
    responses[50, 70] = 0.5
    return gausssmooth(responses, 10)

def get_patch(img, center, sz):
    # crop coordinates
    x0 = round(int(center[0] - sz[0] / 2))
    y0 = round(int(center[1] - sz[1] / 2))
    x1 = int(round(x0 + sz[0]))
    y1 = int(round(y0 + sz[1]))
    # padding
    x0_pad = max(0, -x0)
    x1_pad = max(x1 - img.shape[1] + 1, 0)
    y0_pad = max(0, -y0)
    y1_pad = max(y1 - img.shape[0] + 1, 0)

    # Crop target
    if len(img.shape) > 2:
        img_crop = img[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad, :]
    else:
        img_crop = img[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]

    im_crop_padded = cv2.copyMakeBorder(img_crop, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_REPLICATE)

    # crop mask tells which pixels are within the image (1) and which are outside (0)
    m_ = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    crop_mask = m_[y0 + y0_pad:y1 - y1_pad, x0 + x0_pad:x1 - x1_pad]
    crop_mask = cv2.copyMakeBorder(crop_mask, y0_pad, y1_pad, x0_pad, x1_pad, cv2.BORDER_CONSTANT, value=0)
    return im_crop_padded, crop_mask

def create_epanechnik_kernel(width, height, sigma):
    # make sure that width and height are odd
    w2 = int(math.floor(width / 2))
    h2 = int(math.floor(height / 2))

    [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
    X = X / np.max(X)
    Y = Y / np.max(Y)

    kernel = (1 - ((X / sigma)**2 + (Y / sigma)**2))
    kernel = kernel / np.max(kernel)
    kernel[kernel<0] = 0
    return kernel

def extract_histogram(patch, nbins, weights=None):
    # Note: input patch must be a BGR image (3 channel numpy array)
    # convert each pixel intensity to the one of nbins bins
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    # calculate bin index of a 3D histogram
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins**2  + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :, 2]).astype(np.int32)

    # count bin indices to create histogram (use per-pixel weights if given)
    if weights is not None:
        histogram_ = np.bincount(bin_idxs.flatten(), weights=weights.flatten())
    else:
        histogram_ = np.bincount(bin_idxs.flatten())
    # zero-pad histogram (needed since bincount function does not generate histogram with nbins**3 elements)
    histogram = np.zeros((nbins**3, 1), dtype=histogram_.dtype).flatten()
    histogram[:histogram_.size] = histogram_
    return histogram

def backproject_histogram(patch, histogram, nbins):
    # Note: input patch must be a BGR image (3 channel numpy array)
    # convert each pixel intensity to the one of nbins bins
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    # calculate bin index of a 3D histogram
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins**2  + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :, 2]).astype(np.int32)

    # use histogram as a lookup table for pixel backprojection
    backprojection = np.reshape(histogram[bin_idxs.flatten()], (patch.shape[0], patch.shape[1]))
    return backprojection

def epachenik_derivative_kernel(x):
    return np.ones_like(x)

def mean_shift_alone(image, position, h, kernel=epachenik_derivative_kernel):
    x_k = position[0]
    y_k = position[1]
    min_distance = 1

    position_array = [[x_k, y_k]]
    
    idx = 1
    while min_distance >= 0.001 and idx < 50:
        idx = 1 + idx

        patch = get_patch(image, (x_k, y_k), (h, h))
        l = np.arange(-int(h/ 2), int(h / 2) + 1)
        xi, yi = np.meshgrid(l, l)

        gx = kernel(np.power((x_k - xi)/h, 2))
        gy = kernel(np.power((y_k - yi)/h, 2))

        wx = np.sum(patch[0] * gx)
        wy = np.sum(patch[0] * gy)
        # print(np.sum(patch[0]))
        x_shift = np.sum(np.multiply(np.multiply(patch[0],  xi), gx)) / wx
        y_shift = np.sum(patch[0] * yi * gy) / wy


        # print("x_shift: ", x_shift)
        # print("y_shift: ", y_shift)
        # print("x_k: ", x_k)
        # print("y_k: ", y_k)

        x_k = x_k + x_shift
        y_k = y_k + y_shift

        # print("difference: ", np.sqrt((x_shift)**2 + (y_shift)**2))
        min_distance = np.sqrt((x_shift)**2 + (y_shift)**2)
        position_array.append([x_k, y_k])
    
    # print(len(position_array))
    return (x_k, y_k), position_array

if __name__ == '__main__':
    responses = generate_responses_1()* 255

    # cv2.imshow('responses', responses)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # pos, pos_array = mean_shift_alone(responses, (120, 120), 31)
    # print(pos)
    # print(pos_array)

    # for pos in pos_array:
        # cv2.circle(responses, (int(pos[0]), int(pos[1])), 1, (255, 0, 0), -1)
    
    # cv2.imshow('responses', responses)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    responses = np.repeat(responses[:, :, np.newaxis], 3, axis=2)
    # hist = extract_histogram(responses, 16)
    hist = extract_histogram(np.random.randint(0, 255, (100, 100, 3)), 16)
    print(hist)
    print(len(hist))
    print(np.sum(hist/ np.sum(hist)))

# base class for tracker
class Tracker():
    def __init__(self, params):
        self.parameters = params

    def initialize(self, image, region):
        raise NotImplementedError

    def track(self, image):
        raise NotImplementedError
