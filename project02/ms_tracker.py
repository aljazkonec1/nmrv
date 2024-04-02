import numpy as np
import cv2
from matplotlib import pyplot as plt
from ex2_utils import Tracker
from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram, backproject_histogram
import ex1_utils

class MSParams():
    def __init__(self):
        self.enlarge_factor = 1.5
        self.kernel_size = 15
        self.n_bins = 16
        self.alpha = 0.05


class MeanShiftTracker(Tracker):

    def __init__(self, params):
        self.parameters = params

    def bbox_calculation(self, center, size ): # region format = [x, y, w, h]
        x0 = max(round(int(center[0] - size[0] / 2)), 0)
        y0 = max(round(int(center[1] - size[1] / 2)), 0)

        return x0, y0

    def make_odd(self, val):
        v = np.ceil(val)
        if v % 2 == 0:
            return v - 1
        else: 
            return v
        
    def tl_br(self, region, s): #convert x,y, w, h, to x0, y0, x1, y1 (in preveri da je dolzna regiona liha)
        region = [int(i) for i in region]

        x0 = max(region[0], 0)
        y0 = max(region[1], 0)

        x1 = min(region[0] + self.make_odd(region[2]), s[1] - 1)
        y1 = min(region[1] + self.make_odd(region[3]), s[0] - 1)

        return x0, y0, x1, y1

        
    def initialize(self, image, region):
        
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
        
        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        
        x0, y0, x1, y1 = self.tl_br(region, image.shape)

        self.position = (int(region[0] + region[2] / 2), int(region[1] + region[3] / 2))
        self.size = (self.make_odd(region[2]),self.make_odd(region[3]))

        t = image[int(y0):int(y1), int(x0):int(x1)]
        # t = np.floor(t * (self.parameters.n_bins - 1) / 255)
        self.weights = create_epanechnik_kernel(self.size[0], self.size[1],  self.parameters.kernel_size)

        self.nc = np.sum(self.weights)
        q = extract_histogram(t, self.parameters.n_bins, self.weights[0:t.shape[0], 0:t.shape[1]])

        self.q = self.nc * q / (self.parameters.n_bins** 3)

    def epanechnik_derivative_kernel(self, x):
        return np.ones_like(x)

    def mean_shift(self, image, h,kernel = epanechnik_derivative_kernel): #input is pdf search region
        x_k = self.position[0]
        y_k = self.position[1]
        min_distance = 1

        position_array = [[x_k, y_k]]
        
        idx = 1
        while min_distance >= 0.001 and idx < 50:
            idx = 1 + idx

            patch = get_patch(image, (x_k, y_k), (h, h))
            l = np.arange(-int(h/2), int(h/2) + 1)
            xi, yi = np.meshgrid(l, l)

            gx = kernel(np.power((x_k - xi)/h, 2))
            gy = kernel(np.power((y_k - yi)/h, 2))
            wx = np.sum(patch[0] * gx)
            wy = np.sum(patch[0] * gy)
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

        self.position = (x_k, y_k) # new center
        # print(len(position_array))
        # return (x_k, y_k), position_array

    def track(self, image):

        #mean shift iterations
        x_k = self.position[0]
        y_k = self.position[1]
        min_distance = 1
        position_array = [[x_k, y_k]]
        l = np.arange(-int(self.size[0]/2), int(self.size[0]/2) + 1)
        k = np.arange(-int(self.size[1]/2), int(self.size[1]/2) + 1)
        xi, _ = np.meshgrid(l, k)
        yi, _ = np.meshgrid(k, l)
        yi = yi.T

        idx = 1
        while min_distance >= 0.001 and idx<50:
            idx = 1 + idx

            #to get weights
            nf = get_patch(image, (x_k, y_k), self.size)[0]
            p = extract_histogram(nf, self.parameters.n_bins, self.weights[0:nf.shape[0], 0:nf.shape[1]])
            p = self.nc * p / (self.parameters.n_bins** 3)
            v = np.sqrt(np.divide(self.q, p + 1e-10))
            wi = backproject_histogram(nf, v, self.parameters.n_bins)
            w = np.sum(wi)

            x_shift = np.sum(np.multiply(wi, xi)) / w
            y_shift = np.sum(np.multiply(wi, yi)) / w

            x_k = x_k + x_shift
            y_k = y_k + y_shift

            min_distance = np.sqrt((x_shift)**2 + (y_shift)**2)
            position_array.append([x_k, y_k])

        # update model
        self.position = (x_k, y_k)
        self.q = self.q *(1 - self.parameters.alpha) + self.parameters.alpha * p
        x, y = self.bbox_calculation(self.position, self.size)
        return [x, y, self.size[0], self.size[1]]

        