import numpy as np
import cv2
import ex1_utils as utils
import matplotlib.pyplot as plt
import time

def better_derivatives(im1, im2):
    Dx1, Dy1 = utils.gaussderiv(np.float64(im1), 1.5 )
    Dx2, Dy2 = utils.gaussderiv(np.float64(im2), 1.5)

    Dx = (Dx1 + Dx2)/2
    Dy = (Dy1 + Dy2)/2

    return Dx, Dy, utils.gausssmooth( im2 - im1, 1.5)

def lucas_kanade(im1, im2, N, harris=False, threshold_harris=100):
    kernel = np.ones((N, N))
    
    Dx, Dy, Dt = better_derivatives(im1, im2)

    Dx2 = np.power(Dx, 2)
    Dy2 = np.power(Dy, 2)
    DxDy = np.multiply(Dx, Dy)


    Dx2_sum = cv2.filter2D(Dx2, -1, kernel)
    Dy2_sum = cv2.filter2D(Dy2, -1, kernel)
    DxDy_sum = cv2.filter2D(DxDy, -1, kernel)
    DxDT_sum = cv2.filter2D(np.multiply(Dx ,Dt), -1, kernel)
    DyDT_sum = cv2.filter2D(np.multiply(Dy,Dt), -1, kernel)

    D = np.multiply(Dx2_sum, Dy2_sum) - np.power(DxDy_sum, 2)
    D[D==0] = 1e-8

    U = -(Dy2_sum * DxDT_sum - DxDy_sum * DyDT_sum) / D
    V = -(Dx2_sum * DyDT_sum - DxDy_sum * DxDT_sum) / D

    if harris:
        
        Sx2 = utils.gausssmooth(Dx2, 1.5)
        Sy2 = utils.gausssmooth(Dy2, 1.5)
        Sxy = utils.gausssmooth(DxDy, 1.5)
        # Sx2 = Dx2
        # Sy2 = Dy2
        # Sxy = DxDy

        det_M = (Sx2 * Sy2 - np.power(Sxy, 2))
        # print("det_M", det_M)

        trace_M = (Sx2 + Sy2)
        # print(trace_M)

        harris_response = det_M - 0.05 * np.power(trace_M, 2)
        # print(np.abs(harris_response))
        haris_mask = np.abs(harris_response) > threshold_harris
        # print(haris_mask)
        # print(np.unique(haris_mask, return_counts=True))

        U = np.multiply(U, haris_mask)
        V = np.multiply(V, haris_mask)


    return U, V

def horn_schunck(im1, im2, n_iters, lbbd):
    U = np.zeros(im1.shape)
    V = np.zeros(im1.shape)
    U, V = lucas_kanade(im1, im2, 3)

    Dx, Dy, Dt = better_derivatives(im1, im2)
    Dt = utils.gausssmooth( im2 - im1, 1.5)


    L_d = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])

    P = Dt
    D = np.power(Dx, 2) + np.power(Dy, 2) + lbbd
    D[D==0] = 1e-8

    for i in range(n_iters):
        ua = cv2.filter2D(U, -1, L_d)
        va = cv2.filter2D(V, -1, L_d)
        
        P = Dt + Dx * ua + Dy * va

        U_new = ua - Dx * (P/D)

        if np.linalg.norm(U - U_new) < 1e-8:
            print( np.linalg.norm(U - U_new))
            print("Converged")
            return U, V
        

        U = ua - Dx * (P/D)
        V = va - Dy * (P/D)
        
    return U, V

if __name__ == "__main__":


    im1 = cv2.imread('project01/collision/00000001.jpg', cv2.IMREAD_GRAYSCALE)
    # N=3
    im2 = cv2.imread('project01/collision/00000002.jpg', cv2.IMREAD_GRAYSCALE)
    # im1 = np.random.rand( 200, 200 ).astype(np.float32 )

    # im2 = im1.copy()
    # im2 = utils.rotate_image( im2, -1)
    # times_Lk = []
    # for i in range(500):
    #     print(i)
    #     t = time.time()
    #     U_lk , V_lk = lucas_kanade( im1 , im2 , 3, harris=False)
    #     times_Lk.append(np.round(time.time() - t, 3))
        
    times_hs = []
    for i in range(10):
        print(i)
        t = time.time()
        U_hs , V_hs = horn_schunck(im1 , im2 , 10000 , 1 )
        a = np.round(time.time() - t, 3)
        print(a)
        times_hs.append(a)
    
    # print("Lucas-Kanade: ", np.mean(times_Lk))
    print("Horn-Schunck: ", np.mean(times_hs))
                     
    # U_lk , V_lk = horn_schunck(im1 , im2 , 100 , 2 )

    # fig1 , ( ( ax1_11 , ax1_12 ) , ( ax1_21 , ax1_22 ) ) = plt.subplots ( 2 , 2 )
    # ax1_11.imshow(im1, cmap='gray')
    # ax1_12.imshow(im2,  cmap='gray')

    # utils.show_flow( U_lk , V_lk , ax1_21 , type='angle' )
    # utils.show_flow( U_lk , V_lk , ax1_22 , type='field' , set_aspect=True )
    # fig1.suptitle('Lucas-Kanade O p ti c al Flow ' )

    # fig2, ( ( ax2_11 , ax2_12 ) , ( ax2_21 , ax2_22 ) ) = plt.subplots(2 , 2)
    # ax2_11.imshow(im1, cmap='gray')
    # ax2_12.imshow(im2, cmap='gray')

    # # U_hs , V_hs = horn_schunck(im1 , im2 , 1000 , 0.5 )
    # U_hs , V_hs = lucas_kanade( im1 , im2 , 3, harris=True)

    # utils.show_flow( U_hs,  V_hs , ax2_21 , type='angle' )
    # utils.show_flow( U_hs , V_hs , ax2_22 , type='field' , set_aspect=True )
    # fig2.suptitle ( 'Horn−Schunck O p ti c al Flow ' )

    # plt.show ( )
