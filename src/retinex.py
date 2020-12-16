import numpy as np
import cv2
from skimage import img_as_float64

def singleScale(img,sigma):
    """
    Single-scale Retinex
    
    Parameters :

    img : input image
    sigma : the standard deviation in the X and Y directions, for Gaussian filter
    """

    ssr = np.log10(img) - np.log10(cv2.GaussianBlur(img,(0,0),sigma))
    return ssr

def multiScale(img,sigmas : list):
    """
    Multi-scale Retinex
    
    Parameters :

    img : input image
    sigma : list of all standard deviations in the X and Y directions, for Gaussian filter
    """

    retinex = np.zeros_like(img)
    for s in sigmas:
        retinex += singleScale(img, s)

    msr = retinex/len(sigmas)
    return msr

def crf(img, alpha, beta):
    """
    CRF (Color restoration function)

    Parameters :

    img : input image
    alpha : controls the strength of the nonlinearity
    beta : gain constant
    """
    img_sum = np.sum(img,axis=2,keepdims=True)

    color_rest = beta * (np.log10(alpha*img) - np.log10(img_sum))
    return color_rest


def MSRCR(img, sigmas :list, alpha, beta, G, b):
    """
    MSRCR (Multi-scale retinex with color restoration)

    Parameters :

    img : input image
    sigmas : list of all standard deviations in the X and Y directions, for Gaussian filter
    alpha : controls the strength of the nonlinearity
    beta : gain constant
    G : final gain
    b : offset
    """
    img = img_as_float64(img)+1

    img_msr = multiScale(img, sigmas)    
    img_color = crf(img, alpha, beta)    
    img_msrcr = G * (img_msr*img_color + b)
    
    
    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                                (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                                255

    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    
    return img_msrcr