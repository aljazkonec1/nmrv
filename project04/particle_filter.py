import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from ex4_utils import kalman_step, gaussian_prob, sample_gauss
from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram
# from kalman_filter import NCV, RW, NCA

# from utils.tracker import Tracker

from ex2_utils import Tracker


class ParticleTracker(Tracker): #NCV model

    def __init__(self, n_particles=10, alpha=0.125):
        self.n_particles = n_particles
        self.alpha = alpha
        self.kernel_size = 10
        self.n_bins = 16

        self.A = np.array([[1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
        

    def name(self):
        return "particle"
    
    def make_odd(self, val):
        v = np.ceil(val)
        if v % 2 == 0:
            return v - 1
        else: 
            return v
        
    def bbox_calc(self, region): # input region format = [x, y, w, h], output region format = [x0, y0, x1, y1]
        x, y, w, h = region
        x0 = int(x)
        y0 = int(y)
        x1 = int(x0 + self.make_odd( w + math.modf(region[0])[0]))
        y1 = int(y0 + self.make_odd( h + math.modf(region[1])[0]))

        return [x0, y0, x1, y1]


    def initialize(self, image, region):
        
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x,y, w, h = region

        self.center = (int(x + w/2), int(y + h/2)) # true x, y
        self.region = self.bbox_calc(region) #true x0, y0, x1, y1
        self.size = (self.make_odd( w + math.modf(region[0])[0]), self.make_odd( h + math.modf(region[1])[0])) 
        
        self.region = [int(x) for x in self.region]
        x0, y0, x1, y1 = self.region
        # t = image[y0:y1, x0:x1]  #template

        t = get_patch(image, self.region)
        t = np.array(t[0]) * np.array(t[1][:, :,np.newaxis])

        self.epanechnik_weights = create_epanechnik_kernel(self.size[0], self.size[1],  self.kernel_size)

        self.nc = np.sum(self.epanechnik_weights)
        h = extract_histogram(t, self.n_bins, self.epanechnik_weights)
        self.h = h
        self.h =h / np.sum(h)

        #scale Q_i glede na traget velikost
        q = (self.size[0] + self.size[1]) / 16
        # q = 100
        self.Q_i = np.array([[1/3, 0, 1/2, 0],
                        [0, 1/3, 0, 1/2],
                        [1/2, 0, 1, 0],
                        [0, 1/2, 0, 1]], dtype=np.float32) * q

        # particle generation
        self.particles = sample_gauss((self.center[0], self.center[1], 0, 0), self.Q_i, self.n_particles)
        self.weights = np.ones(self.n_particles) 
        self.weights = self.weights / np.sum(self.weights)



    def track(self, image):
        weights_cumsumed = np.cumsum(self.weights)
        print("weights_cumsumed: ", weights_cumsumed)
        rand_samples = np.random.rand(self.n_particles, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        print("sampled idx", sampled_idxs)
        sampled_particles = self.particles[sampled_idxs.flatten(), :].reshape(-1, 4)
        print("sampled_particles: ", sampled_particles)
        self.particles = self.A.dot(sampled_particles.T).T + sample_gauss((0, 0, 0, 0), self.Q_i, self.n_particles)
        print("particles: ", self.particles)

        # recalculate particle weights
        for i in range(self.n_particles):
            x = self.particles[i][0]
            y = self.particles[i][1]
            # x0 = int(max(x - self.size[0]//2, 0))
            # y0 = int(max(y - self.size[1]//2, 0))
            # x1 = int(min(x + self.size[0]//2, image.shape[1] - 1))
            # y1 = int(min(y + self.size[1]//2, image.shape[0] - 1))
            
            region = self.bbox_calc([int(x - self.size[0]/2), int(y - self.size[1]/2), self.size[0], self.size[1]])
            t = get_patch(image, region)
            patch = np.array(t[0]) * np.array(t[1][:, :,np.newaxis])
            
            # patch = image[y0:y1, x0:x1]

            h = extract_histogram(patch, self.n_bins, self.epanechnik_weights)
            h = h / np.sum(h)
            dist = 1/2 * np.sum((np.sqrt(h) - np.sqrt(self.h))**2)
            # dist =  - np.log(np.sum(np.sqrt(h * self.h)))
            print("dist: ", dist)
            self.weights[i] = np.exp(-1/2 * dist/ 1) #sigma**2 = 100

        print("weights: ", self.weights)
        self.weights = self.weights / np.sum(self.weights)
        #new state of the object
        state = np.sum(self.particles * self.weights[:, np.newaxis], axis=0)
        t_x0 = int(state[0] - self.size[0]//2)
        t_y0 = int(state[1] - self.size[1]//2)

        # cv2.rectangle(image, (t_x0, t_y0), (int(t_x0 + self.size[0]), int(t_y0 + self.size[1])), (0, 255, 0), 2)
        # cv2.imshow('image', image)
        fig, ax = plt.subplots(1, 2)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax[0].imshow(img)
        ax[0].scatter(self.particles[:, 0], self.particles[:, 1], c=self.weights, s=10, cmap='viridis')

        bbox = [t_x0, t_y0, int(self.size[0]), int(self.size[1])]

        # patch = image[t_y0:int(t_y0 + self.size[1]), t_x0:int(t_x0 + self.size[0])]

        region = self.bbox_calc(bbox)
        t = get_patch(image, region)
        patch = np.array(t[0]) * np.array(t[1][:, :,np.newaxis], dtype=np.uint8)
        ax[1].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        plt.show()

        final_h = extract_histogram(patch, self.n_bins, self.epanechnik_weights)
        final_h = final_h / np.sum(final_h)

        self.h = (1- self.alpha) * self.h + self.alpha * final_h

        return bbox

if __name__ == '__main__':
    image = cv2.imread('tst_image.jpg')
    region = [100, 100, 200, 130]
    ncv = ParticleTracker(10, 0.05)
    ncv.initialize(image, region)
    ncv.track(image)
