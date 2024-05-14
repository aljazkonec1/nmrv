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

    def __init__(self, q=100, r= 1):
        T, Q = sp.symbols('T Q')
        
        Fi = sp.Matrix([[1, 0,T, 0],
                        [0, 1, 0, T],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        # print("Fi: ", Fi)
        L = sp.Matrix([[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]])

        Q_i = sp.integrate((Fi * L) * Q * (Fi * L).T, (T, 0, T))
        print("Q_i", Q_i)
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
    def __init__(self, q=100, r =1):
        T, Q = sp.symbols('T Q')
        F = sp.Matrix([[0, 0],[0, 0]])
        L = sp.Matrix([[1, 0], 
                        [0, 1]])


        Fi = np.array(sp.exp(F * T))
        # print("Fi: ", Fi)

        Q_i = sp.integrate((Fi * L) * Q * (Fi * L).T, (T, 0, T))

        # print("Q_i", Q_i)
        self.Q_i = np.array(Q_i.subs({T:1, Q:q})).astype(np.float32)


        self.A = np.eye(2, dtype=np.float32)
        self.C = np.array([[1, 0],
                            [0, 1]], dtype=np.float32)

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
        # print("Fi: ", Fi)
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


        self.Q_i = np.array(Q_i.subs({T:1, Q:1})).astype(np.float32)

        print("Q_i", self.Q_i)

        self.C = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        
        self.R_i = r* np.array([[1, 0],
                    [0, 1]], dtype=np.float32)



if __name__ == '__main__':
    N = 40


    v = np.linspace(5 * math.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v

    nca = NCA()

    # fig, ax = plt.subplots(3, 5, figsize=(20, 10))

    # qs = [100, 5, 1, 1, 1]
    # rs = [1, 1, 1, 5, 100]

    
    

    # for i in range(5):
    #     q = qs[i]
    #     r = rs[i]
    #     ncv = NCV(q, r)
    #     sx, sy = ncv.filter(x, y)
    #     ax[1, i].plot(x, y, 'r')
    #     ax[1, i].plot(sx, sy, 'b')
    #     ax[1, i].scatter(x, y, edgecolors='r', facecolors='none', linewidths=1, s=60)
    #     ax[1, i].scatter(sx, sy, edgecolors='b', facecolors='none', linewidths=1, s=60)
    #     ax[1, i].set_title(f"NCV: q={q}, r={r}")

    #     rw = RW(q, r)
    #     sx, sy = rw.filter(x, y)
    #     ax[0, i].plot(x, y, 'r')
    #     ax[0, i].plot(sx, sy, 'b')
    #     ax[0, i].scatter(x, y, edgecolors='r', facecolors='none', linewidths=1, s=60)
    #     ax[0, i].scatter(sx, sy, edgecolors='b', facecolors='none', linewidths=1, s=60)
    #     ax[0, i].set_title(f"RW: q={q}, r={r}")

    #     nca = NCA(q, r)
    #     sx, sy = nca.filter(x, y)
    #     ax[2, i].plot(x, y, 'r')
    #     ax[2, i].plot(sx, sy, 'b')
    #     ax[2, i].scatter(x, y, edgecolors='r', facecolors='none', linewidths=1, s=60)
    #     ax[2, i].scatter(sx, sy, edgecolors='b', facecolors='none', linewidths=1, s=60)
    #     ax[2, i].set_title(f"NCA: q={q}, r={r}")

    
    # plt.tight_layout()
    # plt.savefig("report/figures/kalman_filter.pdf")



    # # print("nca")
    # # nca = NCA()
    

    # sx, sy = ncv.filter(x, y)
    # plt.plot(x, y, 'r')
    # plt.plot(sx, sy, 'b')
    # plt.scatter(x, y, edgecolors='r', facecolors='none', linewidths=1, s=60)
    # plt.scatter(sx, sy, edgecolors='b', facecolors='none', linewidths=1, s=60)
    # plt.show()


    
    # sx, sy = ncv.filter(x, y)
    # plt.plot(x, y, 'r')
    # plt.plot(sx, sy, 'b')
    # plt.scatter(x, y, edgecolors='r', facecolors='none', linewidths=1, s=60)
    # plt.scatter(sx, sy, edgecolors='b', facecolors='none', linewidths=1, s=60)
    # plt.show()


    # jagged
    # x_jagged = np.linspace(0, 20, N)
    # y_jagged = np.mod(x, 4) + np.random.normal(0, 0.5, x.shape)

    # # on rectangle
    # x_rect = np.arange(0, N, 1)
    # x_rect = np.append(x_rect, [N * np.ones(N), np.arange(N, 0, -1), np.zeros(N)])
    # y_rect = N * np.ones(N)
    # y_rect = np.append(y_rect, [np.arange(N, 0, -1), np.zeros(N), np.arange(0, N, 1)])

    # qs = [50, 1]
    # rs = [1, 50]

    
    # fig, ax = plt.subplots(3, 4, figsize=(20, 10))
    

    # for i in range(2):
    #     q = qs[i]
    #     r = rs[i]

    #     ncv = NCV(q, r)
    #     #jagged
    #     sx, sy = ncv.filter(x_jagged, y_jagged)
    #     ax[1, i].plot(x_jagged, y_jagged, 'r')
    #     ax[1, i].plot(sx, sy, 'b')
    #     ax[1, i].scatter(x_jagged, y_jagged, edgecolors='r', facecolors='none', linewidths=1, s=60)
    #     ax[1, i].scatter(sx, sy, edgecolors='b', facecolors='none', linewidths=1, s=60)
    #     ax[1, i].set_title(f"NCV: q={q}, r={r}")

    #     #rectangle
    #     sx, sy = ncv.filter(x_rect, y_rect)
    #     ax[1, i+2].plot(x_rect, y_rect, 'r')
    #     ax[1, i+2].plot(sx, sy, 'b')
    #     ax[1, i+2].scatter(x_rect, y_rect, edgecolors='r', facecolors='none', linewidths=1, s=60)
    #     ax[1, i+2].scatter(sx, sy, edgecolors='b', facecolors='none', linewidths=1, s=60)
    #     ax[1, i+2].set_title(f"NCV: q={q}, r={r}")

    #     rw = RW(q, r)
    #     #jagged
    #     sx, sy = rw.filter(x_jagged, y_jagged)
    #     ax[0, i].plot(x_jagged, y_jagged, 'r')
    #     ax[0, i].plot(sx, sy, 'b')
    #     ax[0, i].scatter(x_jagged, y_jagged, edgecolors='r', facecolors='none', linewidths=1, s=60)
    #     ax[0, i].scatter(sx, sy, edgecolors='b', facecolors='none', linewidths=1, s=60)
    #     ax[0, i].set_title(f"RW: q={q}, r={r}")

    #     #rectangle
    #     sx, sy = rw.filter(x_rect, y_rect)
    #     ax[0, i+2].plot(x_rect, y_rect, 'r')
    #     ax[0, i+2].plot(sx, sy, 'b')
    #     ax[0, i+2].scatter(x_rect, y_rect, edgecolors='r', facecolors='none', linewidths=1, s=60)
    #     ax[0, i+2].scatter(sx, sy, edgecolors='b', facecolors='none', linewidths=1, s=60)
    #     ax[0, i+2].set_title(f"RW: q={q}, r={r}")


    #     nca = NCA(q, r)
    #     #jagged
    #     sx, sy = nca.filter(x_jagged, y_jagged)
    #     ax[2, i].plot(x_jagged, y_jagged, 'r')
    #     ax[2, i].plot(sx, sy, 'b')
    #     ax[2, i].scatter(x_jagged, y_jagged, edgecolors='r', facecolors='none', linewidths=1, s=60)
    #     ax[2, i].scatter(sx, sy, edgecolors='b', facecolors='none', linewidths=1, s=60)
    #     ax[2, i].set_title(f"NCA: q={q}, r={r}")

    #     # rectangle
    #     sx, sy = nca.filter(x_rect, y_rect)
    #     ax[2, i+2].plot(x_rect, y_rect, 'r')
    #     ax[2, i+2].plot(sx, sy, 'b')
    #     ax[2, i+2].scatter(x_rect, y_rect, edgecolors='r', facecolors='none', linewidths=1, s=60)
    #     ax[2, i+2].scatter(sx, sy, edgecolors='b', facecolors='none', linewidths=1, s=60)
    #     ax[2, i+2].set_title(f"NCA: q={q}, r={r}")


    
    # plt.tight_layout()
    # plt.savefig("report/figures/kalman_filter_other.pdf")

