import numpy as np
import cv2
import ex1_utils as utils
import matplotlib.pyplot as plt
def better_derivatives(im1, im2):
    Dx1, Dy1 = utils.gaussderiv(np.float64(im1), 1.5)
    Dx2, Dy2 = utils.gaussderiv(np.float64(im2), 1.5)

    Dx = (Dx1 + Dx2)/2
    Dy = (Dy1 + Dy2)/2

    return Dx, Dy, utils.gausssmooth( im2 - im1, 1.5)

def lucas_kanade(im1, im2, N):
    
    Dx, Dy, Dt = better_derivatives(im1, im2)

    kernel = np.ones((N, N))

    # print(Dx.shape)

    Dx2_sum = cv2.filter2D(np.power(Dx,2), -1, kernel)
    # print(Dx2_sum.shape)
    Dy2_sum = cv2.filter2D(np.power(Dy, 2), -1, kernel)
    DxDy_sum = cv2.filter2D(np.multiply(Dx, Dy), -1, kernel)
    DxDT_sum = cv2.filter2D(np.multiply(Dx ,Dt), -1, kernel)
    DyDT_sum = cv2.filter2D(np.multiply(Dy,Dt), -1, kernel)

    D = np.multiply(Dx2_sum, Dy2_sum) - np.power(DxDy_sum, 2)

    U = -(Dy2_sum * DxDT_sum - DxDy_sum * DyDT_sum) / D
    V = -(Dx2_sum * DyDT_sum - DxDy_sum * DxDT_sum) / D

    return U, V

def horn_schunck(im1, im2, n_iters, lbbd):
    U = np.zeros(im1.shape)
    V = np.zeros(im1.shape)
    
    Dx, Dy, Dt = better_derivatives(im1, im2)

    L_d = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])

    P = Dt
    D = np.power(Dx, 2) + np.power(Dy, 2) + lbbd

    for i in range(n_iters):
        ua = cv2.filter2D(U, -1, L_d)
        va = cv2.filter2D(V, -1, L_d)
        
        P = Dt + Dx * ua + Dy * va

        U = ua - Dx * (P/D)
        V = va - Dy * (P/D)
        
    return U, V

if __name__ == "__main__":


    # im1 = cv2.imread('project01/collision/00000001.jpg', cv2.IMREAD_GRAYSCALE)
    # im2 = cv2.imread('project01/collision/00000002.jpg', cv2.IMREAD_GRAYSCALE)
    # N = 3

    im1 = np.random.rand( 200, 200 ).astype(np.float32 )
    im2 = im1.copy()
    im2 = utils.rotate_image( im2, -1)
    U_lk , V_lk = lucas_kanade( im1 , im2 , 3 )
    fig1 , ( ( ax1_11 , ax1_12 ) , ( ax1_21 , ax1_22 ) ) = plt.subplots ( 2 , 2 )
    ax1_11.imshow(im1)
    ax1_12.imshow(im2)

    utils.show_flow( U_lk , V_lk , ax1_21 , type='angle' )
    utils.show_flow( U_lk , V_lk , ax1_22 , type='field' , set_aspect=True )
    fig1.suptitle('Lucas-Kanade O p ti c al Flow ' )

    fig2, ( ( ax2_11 , ax2_12 ) , ( ax2_21 , ax2_22 ) ) = plt.subplots(2 , 2)
    ax2_11.imshow(im1)
    ax2_12.imshow(im2)

    U_hs , V_hs = horn_schunck(im1 , im2 , 1000 , 0.5 )

    utils.show_flow( U_hs,  V_hs , ax2_21 , type='angle' )
    utils.show_flow( U_hs , V_hs , ax2_22 , type='field' , set_aspect=True )
    fig2.suptitle ( 'Horn−Schunck O p ti c al Flow ' )

    plt.show ( )
