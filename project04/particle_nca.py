import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from ex4_utils import kalman_step, gaussian_prob, sample_gauss
from ex2_utils import get_patch, create_epanechnik_kernel, extract_histogram

# from utils.tracker import Tracker
from ex2_utils import Tracker


class ParticleTrackerNCA(Tracker): #NCV model

    def __init__(self, n_particles=100, alpha=0.05, n_bins=16, sigma=25, q_qucient=4):
        self.n_particles = n_particles
        self.alpha = alpha
        self.kernel_size = 1
        self.n_bins = n_bins
        self.q_qucient = q_qucient
        self.sigma = sigma
        self.A = np.array([[1, 0, 1, 0, 0.5, 0],
                        [0, 1, 0, 1, 0, 0.5],
                        [0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        

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
        
        x,y, w, h = region

        self.center = (int(x + w/2), int(y + h/2)) # true x, y
        self.region = self.bbox_calc(region) #true x0, y0, x1, y1
        self.size = (self.make_odd( w + math.modf(region[0])[0]), self.make_odd( h + math.modf(region[1])[0])) 
        self.region = [int(x) for x in self.region]

        t = get_patch(image, self.region)
        t = np.array(t[0]) * np.array(t[1][:, :,np.newaxis])
        self.epanechnik_weights = create_epanechnik_kernel(self.size[0], self.size[1],  self.kernel_size)

        h = extract_histogram(t, self.n_bins, self.epanechnik_weights)
        self.h =h / np.sum(h)

        q = min(self.size[0], self.size[1]) /4
        self.Q_i = np.array([[0.05, 0, 0.125, 0, 1/6, 0.],
                            [0., 0.05, 0., 0.125 , 0., 1/6],
                            [0.125, 0., 1/3, 0.,0.5, 0. ],
                            [0., 0.125, 0., 1/3, 0., 0.5],
                            [1/6, 0., 0.5, 0., 1., 0.],
                            [0., 1/6, 0., 0.5, 0., 1.]], dtype=np.float32) * q
        
        self.particles = sample_gauss((self.center[0], self.center[1], 0, 0, 0,0), self.Q_i, self.n_particles)

        self.weights = np.ones(self.n_particles) 
        self.weights = self.weights / np.sum(self.weights)



    def track(self, image):
        best_weights = self.weights
        weights_cumsumed = np.cumsum(best_weights)
        rand_samples = np.random.rand(self.n_particles, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        sampled_particles = self.particles[sampled_idxs.flatten(), :].reshape(-1, 6)


        sg = sample_gauss( (0,0,0, 0,0,0), self.Q_i, self.n_particles)
        self.particles = self.A.dot(sampled_particles.T).T + sg


        for i in range(self.n_particles):
            x = self.particles[i][0]
            y = self.particles[i][1]
            
            region = self.bbox_calc([int(x - self.size[0]/2), int(y - self.size[1]/2), self.size[0], self.size[1]])
            t = get_patch(image, region)
            patch = np.array(t[0]) * np.array(t[1][:, :,np.newaxis])
            
            h = extract_histogram(patch, self.n_bins, self.epanechnik_weights)
            h = h / np.sum(h)
            dist = np.sum((np.sqrt(h) - np.sqrt(self.h))**2)
            self.weights[i] = np.exp(-self.sigma * dist) #sigma**2 = 0.005

        self.weights = self.weights / np.sum(self.weights)

        state = np.sum(self.particles * self.weights[:, np.newaxis], axis=0)
        

        t_x0 = int(state[0] - self.size[0]//2)
        t_y0 = int(state[1] - self.size[1]//2)

        bbox = [t_x0, t_y0, int(self.size[0]), int(self.size[1])]

        region = self.bbox_calc(bbox)
        t = get_patch(image, region)
        patch = np.array(t[0]) * np.array(t[1][:, :,np.newaxis], dtype=np.uint8)

        final_h = extract_histogram(patch, self.n_bins, self.epanechnik_weights)
        final_h = final_h / np.sum(final_h)
        self.h = (1- self.alpha) * self.h + self.alpha * final_h

        return bbox, self.particles[:, 0:2]
