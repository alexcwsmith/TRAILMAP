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

directory='/d2/studies/ClearMap/ROC_SPmR/ROC10/seg-SPmR_cropped/'
#crop = {'width':1900,'height':2320,'x':130,'y':240}
crop=None

def skel(directory, crop=None, flip='y'):
    """Skeletonize TrailMap results.
    
    Parameters
    ----------
    directory : string
        Path to directory with segmented data.
    crop : dict (optional, default None)
        Dictionary with ImageJ-format cropping coordinates ({width:, height:, x:, y:,})
    flip : string (optional, default 'y')
        Option to flip axis, can be any combination of 'xyz'.
    """
    sample = directory.split('/')[-3]
    print("Started " + time.ctime())
    ims = io.ImageCollection(os.path.join(directory, '*.tif'), load_func=io.imread)
    data = ims.concatenate()
    if crop:
        rawshape=data.shape
        data = data[:,crop['y']:crop['y']+crop['height'],crop['x']:crop['x']+crop['width']]
        print("Cropped data from " + str(rawshape) + " to " + str(data.shape) + " at " + time.ctime())
    cat = np.zeros(shape=(data.shape), dtype='float32')
    for i in range(2,10,1):
        print(str(i) + " started at " + time.ctime())
        i=i/10
        im = (data>i).astype('float32')
        skel = morphology.skeletonize_3d(im).astype('float32')*i
        print(str(i) + " completed at " + time.ctime())
        cat = cat+skel
    if 'y' in flip:
        cat = np.flip(cat, axis=1)
    if 'x' in flip:
        cat = np.flip(cat, axis=2)
    if 'z' in flip:
        cat = np.flip(cat, axis=0)
    io.imsave(os.path.join(directory, sample + '_ThresholdedSkeleton3D.tif'), cat, check_contrast=False)
    print("Finished " + sample + ' ' + time.ctime())
    return cat

cat = skel(directory, crop=None, flip=None)



if __name__ == '__main__':
    p = argparse.ArgumentParser
    p.add_argument('--directory',type=str,help='Directory with segmented images.')
    p.add_argument('--crop',type=dict,default=None,help='Dict with ImageJ-format cropping coordinates.')
    p.add_argument('--flip',tyep=str,default=None,help='xyz axes to flip')
    args = p.parse_args()
    directory = args['directory']
    crop = args['crop']
    flip = args['flip']
    skel(directory, crop=crop, flip=flip)
    
    
