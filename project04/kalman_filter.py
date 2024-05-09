import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from ex4_utils import kalman_step
import ex2_utils
import sympy as sp

class KalmanFilter:
    def __init__(self):
        self.A = None
        self.C = None
        self.Q_i = None
        self.R_i = None

    def filter(self, x, y):
        sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
        sy = np.zeros((y.size,1), dtype=np.float32 ).flatten()
        sx[0] = x[0]
        sy[0] = y[0]
        state = np.zeros((self.A.shape[0], 1), dtype=np.float32 ).flatten()
        state [0] = x[0]
        state[1] = y[0]
        covariance = np.eye(self.A.shape[0], dtype=np.float32 ) #P

        for j in range(1, x.size):
            state, covariance , _ , _ = kalman_step(self.A,self.C,self.Q_i,self.R_i,np.reshape(np.array([x[j], y[j]]),(-1, 1)), np.reshape(state, (-1,1)), covariance)
            sx[j] = state[0][0]
            sy[j] = state[1][0]

            # print(state)
        return sx, sy


class NCV(KalmanFilter): #Nearly constant velocity model

    def __init__(self, q=1, r= 100):
        T, Q = sp.symbols('T Q')
        
        Fi = sp.Matrix([[1, 0,T, 0],
                        [0, 1, 0, T],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        L = sp.Matrix([[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]])

        Q_i = sp.integrate((Fi * L) * Q * (Fi * L).T, (T, 0, T))
        # print(Q_i)
        self.Q_i = np.array(Q_i.subs({T:1, Q:q})).astype(np.float32)

        self.A = np.array([[1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
        self.C = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]], dtype=np.float32)

        self.R_i = r* np.array([[1, 0],
                    [0, 1]], dtype=np.float32)
        
        

#random walk model
class RW(KalmanFilter):
    def __init__(self, q=100, r = 1):
        T, Q = sp.symbols('T Q')
        F = sp.Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        L = sp.Matrix([[1, 0], 
                        [0, 1],
                        [0, 0], 
                        [0, 0]])


        Fi = np.array(sp.exp(F * T))
        Q_i = sp.integrate((Fi * L) * Q * (Fi * L).T, (T, 0, T))

        self.Q_i = np.array(Q_i.subs({T:1, Q:q})).astype(np.float32)


        self.A = np.eye(4, dtype=np.float32)
        self.C = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]], dtype=np.float32)

        self.R_i = r* np.array([[1, 0],
                    [0, 1]], dtype=np.float32)
        

class NCA(KalmanFilter): #nearly constant acceleration model

    def __init__(self, q=100, r= 1):
        T, Q = sp.symbols('T Q')

        F = sp.Matrix([[0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]])
        L = sp.Matrix([[0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [1, 0],
                        [0, 1]])
        
        Fi = np.array(sp.exp(F * T))
        # print(Fi)
        # Fi = sp.Matrix([[1, 0,T, 0, 0.5*T**2, 0],
        #                     [0, 1, 0, T, 0, 0.5*T**2],
        #                     [0, 0, 1, 0, T, 0],
        #                     [0, 0, 0, 1, 0, T],
        #                     [0, 0, 0, 0, 1, 0],
        #                     [0, 0, 0, 0, 0, 1]])

        self.A = np.array([[1, 0, 1, 0, 0.5, 0],
                        [0, 1, 0, 1, 0, 0.5],
                        [0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]], dtype=np.float32)
        
        Q_i = sp.integrate((Fi * L) * Q * (Fi * L).T, (T, 0, T))
        print(Q_i)

        self.Q_i = np.array(Q_i.subs({T:1, Q:q})).astype(np.float32)

        self.C = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        
        self.R_i = r* np.array([[1, 0],
                    [0, 1]], dtype=np.float32)
        

if __name__ == '__main__':
    N = 40

    v = np.linspace(5 * math.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v
    ncv = NCV()
    sx, sy = ncv.filter(x, y)
    plt.plot(x, y, 'r')
    plt.plot(sx, sy, 'b')
    plt.show()
