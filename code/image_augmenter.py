#!/usr/bin/env python

import cv2
import numpy as np


################################################################################

def image_rotation(img, angle=90):
    """
    Rotates images with respect to the center of the image.
    Args:
        img ():
    """

    row, col = img.shape
    M_rot    = cv2.getRotationMatrix2D((col/2, row/2), angle, 1)
    new_img  = cv2.warpAffine(img, M_rot, (col, row))

    return new_img


################################################################################

def image_flip(img, horizontal=True, vertical=False):
    """
    Flips image vertically and/or horizontally.
    Args:
        img( ): 2D Numpy Array
        horizontal (bool): Flag to flip image horizontally. [Default=True.]
        vertical (bool): Flag to flip image vertically. [Default=False.]
    Returns
        Flipped image.

    Note: An image that is flipped both vertically and horizontally is equivalent 
          to a 180 degree rotation about the center of the original image.
    """
    if horizontal:
        img = img[:, ::-1]
    if vertical:
        img = img[::-1, :]

    return img

################################################################################

def image_shift(img, pixels=0)
    return img

################################################################################

def image_crop(img, angle=45):
    
    
    return img 

################################################################################
################################################################################

def augment(fname):
    """

    Args:
        fname (str): File name of original image

    Returns:
        List of augmented images
    """
    
    img = cv2.imread(fname)
    
    pass

################################################################################

if __name__ == "__main__":
    augment()

################################################################################
