#! /usr/bin/python
"""
# ==============================
# kittitool.py
# this file provides read/write and visualize functions of optical flow files in kitti format
# Author: Ruoteng Li
# Date: 6th Aug 2016
# ==============================
"""


import png
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt
from lib import flowlib as fl

def remove_ambiguity_flow(src_img, warp_img, flow_img):
    image_height    = src_img.shape[0]
    image_width     = src_img.shape[1]
    src_img     = src_img.astype('int16')
    warp_img    = warp_img.astype('int16')
    err_img     = np.abs(src_img - warp_img)
    err_img     = np.max(err_img, axis = 2)
    # mask_img    = np.ones((src_img.shape[0], src_img.shape[1]))
    flow_img[err_img > 20] = 0.0
    return flow_img
