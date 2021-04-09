#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:14:35 2020

@author: smith
"""

import os
import skimage
from skimage import io
from skimage import morphology
import numpy as np
from scipy import ndimage
import time
import argparse

#directory='/d2/studies/ClearMap/ROC_SPmR/ROC10/seg-SPmR_cropped/'
#crop = {'width':1900,'height':2320,'x':130,'y':240}


def Skeletonize3D(directory, crop=None, flip='y', dtype=None):
    """Skeletonize TrailMap results.
    
    Parameters
    ----------
    directory : string
        Path to directory with segmented data.
    crop : dict (optional, default None)
        Dictionary with ImageJ-format cropping coordinates ({width:, height:, x:, y:,})
    flip : string (optional, default 'y')
        Option to flip axis, can be any combination of 'xyz'.
    dtype : numpy dtype (optional, default None results in float32 images)
        Data type for output image. Set dtype=np.uint16 if you are going to combine with autofluo in Imaris.
    """
    #Load Data:
    sample = directory.split('/')[-3]
    print("Started " + time.ctime())
    ims = io.ImageCollection(os.path.join(directory, '*.tif'), load_func=io.imread)
    data = ims.concatenate()
    #Optionally crop:
    if crop:
        rawshape=data.shape
        data = data[:,crop['y']:crop['y']+crop['height'],crop['x']:crop['x']+crop['width']]
        print("Cropped data from " + str(rawshape) + " to " + str(data.shape) + " at " + time.ctime())
    cat = np.zeros(shape=(data.shape), dtype='float32') #Create output array
    #Loop through thresholds 0.2 -> 0.9, extract signal, scale, and combine
    for i in range(2,10,1):
        print(str(i) + " started at " + time.ctime())
        i=i/10
        im = (data>i).astype('float32')
        skel = morphology.skeletonize_3d(im).astype('float32')*i
        print(str(i) + " completed at " + time.ctime())
        cat = cat+skel
    #Optionally flip along the x, y, or z axis:
    if flip:
        if 'y' in flip:
            cat = np.flip(cat, axis=1)
        if 'x' in flip:
            cat = np.flip(cat, axis=2)
        if 'z' in flip:
            cat = np.flip(cat, axis=0)
    if dtype:
        cat = cat.astype(dtype) #have not tested that this results in same pixel values as changing image type in ImageJ.
    #Save the result image stack:
    try:
        io.imsave(os.path.join(directory, sample + '_ThresholdedSkeleton3D.tif'), cat, check_contrast=False)
    except PermissionError:
        print("You do not have write permissions for " + str(directory) + '\n' + "Saving to your home directory instead.")
        homedir = os.path.expanduser('~/')
        io.imsave(os.path.join(homedir, sample + '_ThresholdedSkeleton3D.tif'), cat, check_contrast=False)
    print("Finished " + sample + ' ' + time.ctime())
    return cat


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--directory',type=str,default=os.getcwd(),help='Directory with segmented images.')
    p.add_argument('--crop',type=dict,default=None,help='Dict with ImageJ-format cropping coordinates.')
    p.add_argument('--flip',type=str,default=None,help='xyz axes to flip')
    p.add_argument('--dtype',type=np.dtype,default=None,help='NumPy dtype to save as')
    args = p.parse_args()
    Skeletonize3D(args.directory, crop=args.crop, flip=args.flip, dtype=args.dtype)
    
    
