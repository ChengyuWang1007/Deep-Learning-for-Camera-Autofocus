#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A library containing functions for depth to defocus

Created on Sat Feb 16 15:12:16 2019
@author: minghao
"""

import cv2
import numpy as np

def cal_disk_radius(f, fn, d_tgt, pp, d_obj):
    """
    This function calculates blur radius with naive ray optic method
    Inputs:
        f - focal length
        fn - f number
        d_tgt - depth of target object (the object exactly in focus)
        pp - pixel pitch
        d_obj - depth of object point
    Outputs:
        r_disk - radius of disk in pixels
    Note: 
        May return positive/negative radius, indicating defocus direction
        Length unit is mm if not specified
    """
    D = f/fn # aperture diameter
    r_disk = D/2*(1.0/d_obj-1.0/d_tgt)/(1/f-1/d_tgt)/pp #disk radius in pixel
    return -r_disk # minus: to make r and d positive correlated

def cal_obj_depth(f, fn, d_tgt, pp, r_disk):
    """
    This function calculates object depth with naive ray optic method
    Inputs:
        f - focal length
        fn - f number
        d_tgt - depth of target object (the object exactly in focus)
        pp - pixel pitch
        r_disk - radius of disk in pixels
    Outputs:
        d_obj - depth of object point
    Note: 
        Accepts negative radius
        Length unit is mm if not specified
    """
    D = f/fn # aperture diameter
    # get a intermediate value, the minus is to make positive correlated
    x_inter = -2*r_disk*pp/D*(1.0/f-1.0/d_tgt) 
    d_obj = 1/(x_inter+1.0/d_tgt)
    return d_obj

def disk_filter(radius):
    """
    This function makes a disk filter with given radius
    Input:
        radius - the radius of disk in pixels
    Output:
        df - disk filter
    """
    assert radius>=0, "no negtive radius!"
    # determine filter size and create container
    r_int = np.floor(radius) # int radius
    xs = np.linspace(-r_int, r_int, 2*np.int(r_int)+1)
    fl = xs.shape[0]
    df = np.zeros((fl, fl))
    # fill disk filter
    x2s = np.power(xs, 2)
    r2 = radius**2
    for a in range(fl):
        for b in range(fl):
            if x2s[a]+x2s[b] <= r2:
                df[a,b] = 1.0
    # normalize
    df = df/np.sum(df)
    return df

def depth2defocus(img, dep, d_tgt, f, fn, pp, *, 
                  gamma=2.2, r_step=0.2, inpaint_occlusion=True):
    """
    This function creates a blurred image based on defocus
    Inputs:
        img - 3d uin8 np array with shape (h,w,c), representing a 2d color image
        dep - 2d float np array with shape (h,w), representing depth
        d_tgt - depth of target object (the object exactly in focus)
        f - focal length
        fn - f number
        pp - pixel pitch
        gamma - positive float scalar for gamma correction
        r_step - the step of disk radii, in pixels
        inpaint_occlusion - boolean, inpaint occlusion or not
    Outputs:
        defocused_image - 3d uin8 np array with shape (h,w,3),
        representing the blurred image based on defocus
    Notes:
        Length unit is mm if not specified
    """
    # get a list of disk filter based on depth
    depth_range = (dep.min(), dep.max())
    r_range = tuple([cal_disk_radius(f, fn, d_tgt, pp, d) for d in depth_range])
    r_multiplier_range = ((np.floor(r_range[0]/r_step-0.5)+0.5), 
                          (np.ceil(r_range[1]/r_step-0.5)+0.5))
    r_delimiters = np.arange(r_multiplier_range[0], # radius delimiters
                             r_multiplier_range[1] + 1, 
                             1).astype(np.float)*r_step
    r_delimiters[0] = r_range[0]
    r_delimiters[-1] = r_range[-1] # in case there's outliers
    dfrs = (r_delimiters[:-1]+r_delimiters[1:])/2 # disk filter radii, mid value
    dfs = [disk_filter(np.abs(r)) for r in dfrs] # disk filters
    
    # get image mask based on depth
    d_delimiters = [cal_obj_depth(f, fn, d_tgt, pp, r) \
                    for r in r_delimiters] # depth delimeters
    masks = [dep<=d for d in d_delimiters[1:]]
    
    # re-gamma correction, cv2.inpaint only accept uint8 3-channel image
    optical_img = img.astype(np.float)/255.0 # normalize
    optical_img = np.power(optical_img, gamma) # re-gamma
    optical_img = np.round(optical_img*255).astype(np.uint8)
    
    # prepare to inpaint layers to fight occlusion
    # It's slow to inpaint whole occluded part for each layer
    # Use dilation to inpaint only the "edges"
    # The first "None" marks nearest layer, no need to inpaint
    dilation_kernels = [None] 
    for r1,r2 in zip(np.abs(dfrs[:-1]), np.abs(dfrs[1:])):
        r = np.max([r1,r2,2])
        k = disk_filter(r)
        dilation_kernels.append((k>0).astype(np.uint8))
    
    # get image layers. If needed, inpaint layers.
    layers = []
    last_m = None #just to fool spyder... without this line, it makes false alarm
    for m,k in zip(masks, dilation_kernels):
        l = np.copy(optical_img)
        if inpaint_occlusion and (k is not None): # not the first layer
            # where to inpaint
            last_m_reverse = np.logical_not(last_m) # reverse last mask
            dilated_reverse = cv2.dilate(last_m_reverse.astype(np.uint8), 
                                         k, iterations=3).astype(np.bool)
            inpaint_m = np.logical_and(last_m, dilated_reverse) #area to inpaint
            # inpaint
            l[last_m] = 0 # make occlusion, then inpaint
            l = cv2.inpaint(l,255*inpaint_m.astype(np.uint8),2,cv2.INPAINT_NS)
        l[np.logical_not(m)] = 0 # remove not needed part
        last_m = m
        layers.append(l)
    
    # blur and add up image layers, from far to near
    c = img.shape[2] #color channel
    defocused_optical = np.zeros(img.shape, dtype=np.float)
    for l,m,k in zip(layers[::-1], masks[::-1], dfs[::-1]): 
        # k here means kernel (f is used)
        # blur layer and mask
        l_blurred = cv2.filter2D(l.astype(np.float), -1, k, 
                                 None, (-1,-1), 0, 
                                 cv2.BORDER_CONSTANT)
        m_blurred = cv2.filter2D(m.astype(np.float), -1, k, 
                                 None, (-1,-1), 0, 
                                 cv2.BORDER_CONSTANT)
        
        # apply a complementary mask to background (layers behind)
        defocused_optical *= np.tile(np.expand_dims(1.0-m_blurred,2),(1,1,c))
        # sum up
        defocused_optical += l_blurred
    
    # gamma correction and post process
    defocused_image = defocused_optical/255.0 # normalize
    defocused_image = np.clip(defocused_image, 0, 1)
    defocused_image = np.power(defocused_image, 1/gamma) # re-gamma
    defocused_image = np.round(defocused_image*255).astype(np.uint8)
    
    return defocused_image

'''
def optical_blur(img, kernel, *, gamma=2.2, out_dtype=None):
    """
    This function convolve image with kernel. Optical means considering gamma.
    Inputs:
        img - 3d np uint8 array with shape (h,w,c), representing a 2d color image
        kernel - 2d float np array
        gamma - positive float scalar for gamma correction. None means no correction.
        out_dtype - np.dtype var, meaning output data type, None means uint8
    Outputs:
        blurred_img - 3d np array with shape (h,w,c), representing a 2d color image
    Notes:
        This function use 0 padding for border
    """
    if out_dtype is None:
        out_dtype = img.dtype
    optical_img = img.astype(np.float)
    # re-gamma correct
    if gamma is not None:
        optical_img /= 255.0 #normalize
        optical_img = np.power(optical_img, gamma)
    # blur
    optical_blurred = cv2.filter2D(optical_img, -1, kernel, #basic parameters
                                   None, (-1,-1), 0,  #place holders
                                   cv2.BORDER_CONSTANT) #very important
    img_blurred = optical_blurred
    # gamma correct
    if gamma is not None:
        img_blurred = np.power(img_blurred, 1.0/gamma)
        img_blurred *= 255.0
    # determine type
    return img_blurred.astype(out_dtype)
'''