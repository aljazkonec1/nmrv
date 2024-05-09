import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.tracker import Tracker

# from ex2_utils import Tracker
from ex2_utils import get_patch
from ex3_utils import create_cosine_window, create_gauss_peak

# class Tracker():
#     def __init__(self, params):
#         self.parameters = params

class CorTracker():
    def __init__(self):
        self.kernel_size = 10
        self.alpha = 0.125
        self.lmbda = 0.6
        self.search_window = 1.1

class CorrelationTracker(Tracker):

    def __init__(self, kernel_size=10, alpha=0.125, lmbda=0.6, search_window=1.1):
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.lmbda = lmbda
        self.search_window = search_window


    def name(self):
        return "correlation"
    
    def preprocess_patch(self, patch):
        patch = np.log1p(patch)
        patch = (patch - np.mean(patch))
        patch = patch / np.linalg.norm(patch)
        patch = patch * self.cosine_window 

        return patch
    
    def make_odd(self, val):
        v = np.ceil(val)
        if v % 2 == 0:
            return int(v - 1)
        else: 
            return int(v)

    def bbox_calc(self, region): # input region format = [x, y, w, h], output region format = [x0, y0, x1, y1]
        x, y, w, h = region

        x0 = int(x)
        y0 = int(y)
        x1 = x0 + self.make_odd( w + math.modf(region[0])[0])
        y1 = y0 + self.make_odd( h + math.modf(region[1])[0])
            
        return [x0, y0, x1, y1]
    

    def initialize(self, image, region):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x,y, w, h = region
        self.true_size = self.make_odd(w + math.modf(x)[0]), self.make_odd(h + math.modf(y)[0]) #true w, h
        self.center = (int(x + w/2), int(y + h/2)) # true x, y
        self.region = self.bbox_calc(region) #true x0, y0, x1, y1

        sw = self.search_window
        self.localization_size = (self.make_odd(self.true_size[0] * sw), self.make_odd(self.true_size[1] * sw)) # enlarged w, h
        l_x0 = int(np.ceil(self.center[0] - self.localization_size[0]/2))
        l_y0 = int(np.ceil(self.center[1] - self.localization_size[1]/2))

        self.localization_region = self.bbox_calc([l_x0, l_y0, self.localization_size[0], self.localization_size[1]]) #x0, y0, x1, y1


        patch = get_patch(image, self.localization_region)
        patch = np.array(patch[0]) * np.array(patch[1])


        self.cosine_window = create_cosine_window(self.localization_size)
        self.gauss_function = create_gauss_peak(self.localization_size, self.kernel_size)


        # preprocessing of patch
        patch = self.preprocess_patch(patch)

        patch = np.roll(patch, (-patch.shape[0]//2, -patch.shape[1]//2), (0,1))
        patch_fft = np.fft.fft2(patch)

        # self.gauss_fft = np.fft.fft2(self.gauss_function)
        self.gauss_fft = self.gauss_function

        con_f = np.conj(patch_fft)

        self.filter_H = (np.multiply(self.gauss_fft, con_f)) / ( patch_fft * con_f + self.lmbda)
    
    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        patch = get_patch(image, self.localization_region)
        patch = np.array(patch[0]) * np.array(patch[1])
        

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(patch, cmap='gray')

        patch = self.preprocess_patch(patch)
        patch = np.roll(patch, (-patch.shape[0]//2, -patch.shape[1]//2), (0,1))

        patch = np.fft.fft2(patch)

        response = np.real(np.fft.ifft2(self.filter_H * patch))
        response = np.roll(response, (patch.shape[0]//2, patch.shape[1]//2), (0,1))


        y,x= np.unravel_index(np.argmax(response), response.shape)
        # print("")
        # print("image shape: ", image.shape)
        # print("max response: ", y, x )

        # r = response / max(response.flatten())

        # cv2.imshow('response', r)
        

        x0 =self.localization_region[0]
        y0 =self.localization_region[1]

        # print("localization region ", self.localization_region)
        # print("localization region shape ", response.shape)

        # print("center ", self.center)
        self.center = (int(x + x0), int(y + y0)) # (x_center, y_center)
        # print("after update center ", self.center)

        t_x0 = int(x + x0 - self.true_size[0]/2)
        t_y0 = int(y + y0 - self.true_size[1]/2)

        bbox = [t_x0, t_y0, self.true_size[0], self.true_size[1]] # true x0 , y0, w, h

        self.region = self.bbox_calc(bbox) #true x0, y0, x1, y1

        l_x0 = int(np.ceil(self.center[0] - self.localization_size[0]/2))
        l_y0 = int(np.ceil(self.center[1] - self.localization_size[1]/2))
        self.localization_region = self.bbox_calc([l_x0, l_y0, self.localization_size[0], self.localization_size[1]]) #local x0, y0, x1, y1
        
        if self.localization_region[0] > image.shape[1] or self.localization_region[1] > image.shape[0] or self.localization_region[2] < 0 or self.localization_region[3] < 0:
            return [0, 0, 0, 0]
        
        target = get_patch(image, self.region)
        target = np.array(target[0]) * np.array(target[1])


        # cv2.imshow('target', target/255)

        localization = get_patch(image, self.localization_region)
        localization = np.array(localization[0]) * np.array(localization[1])

        # cv2.imshow('localization', localization/255)

        patch = get_patch(image, self.localization_region)

        patch = np.array(patch[0]) * np.array(patch[1])

        patch = self.preprocess_patch(patch)
        patch = np.roll(patch, (-patch.shape[0]//2, -patch.shape[1]//2), (0,1))
        patch = np.fft.fft2(patch)

        H = (self.gauss_fft * np.conj(patch)) / (patch * np.conj(patch) + self.lmbda)

        self.filter_H = (1 - self.alpha) * self.filter_H + self.alpha * H

        return bbox
