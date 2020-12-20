import os
import cv2 
import argparse
from skimage import img_as_float64
import matplotlib.pyplot as plt


def read_show(file_path,file,show=False):
    '''
    A function to read an image and return a RGB version of it,
    also display the image.

    Args:

    file_path : the path/location of the input file
    file :  input filename   
    show :  if set "True" displays the image
    '''

    img = cv2.imread(os.path.join(file_path,file))
    b,g,r = cv2.split(img)       # get b,g,r
    img = cv2.merge([r,g,b])     # switch it to rgb
    
    if show==True:
        plt.imshow(img)
        plt.xticks([]),plt.yticks([])
        plt.show()
    return img


def plot_hist(orig_img, enh_img, hist=False, save=False, fname=None):
    '''
    A function to display original and enhanced images.


    Args:

    origin_img : input image
    enh_img : MSRCR output  
    save : FALSE(default); Set it to TRUE to save the output to assets.
    '''
    if hist :
        fig, ax = plt.subplots(2,2, figsize=(20,15))

        ax[0,0].imshow(orig_img)
        ax[0,0].set_xticks([])
        ax[0,0].set_yticks([])
        ax[0,0].set_title('Original',fontsize=25)
        ax[1,0].hist(orig_img.ravel(),256,[0,256])

        ax[0,1].imshow(enh_img)
        ax[0,1].set_xticks([])
        ax[0,1].set_yticks([])
        ax[0,1].set_title('Enhanced',fontsize=25)
        ax[1,1].hist(enh_img.ravel(),256,[0,256])

        fig.suptitle('Multi-scale retinex  with color restoration', fontsize=30, y=1.05)
        fig.tight_layout()

    else :
        fig, ax = plt.subplots(1,2, figsize=(15,8))

        ax[0].imshow(orig_img)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title('Original',fontsize=25)
        ax[1].imshow(enh_img)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title('Enhanced',fontsize=25)

        fig.suptitle('Multi-scale retinex  with color restoration', fontsize=30, y=1.05)
        fig.tight_layout()

    
    if save:
        if fname is not None:
            save_file = os.path.join('assets', fname) 
            plt.savefig(save_file,bbox_inches='tight',dpi=72)
            plt.close(fig)

    

def checker(path):
    if os.path.basename is None:
        raise argparse.ArgumentTypeError('File name not included in the path.')

    if os.path.basename not in os.path.dirname:
        raise argparse.ArgumentTypeError(f'{os.path.basename} doesn\'t exist in given path')
    return