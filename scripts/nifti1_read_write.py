#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:44:21 2020

@author: andy
"""

import ctypes
from ctypes import c_char_p
from ctypes import c_float
from ctypes import c_int
#from ctypes import Array
from ctypes import byref
#from ctypes import pointer
import time
import numpy as np
import matplotlib.pyplot as plt

#fun = ctypes.CDLL('/home/andy/projects/lukasSegmenter/biomedical_segmenter/scripts/my_nifti1_read.so')
fun = ctypes.CDLL('/home/andy/projects/lukasSegmenter/biomedical_segmenter/scripts/my_nifti1_read_patch.so')

fun.main.restype = ctypes.POINTER(ctypes.c_float)
#fun.main()
#fun.main('./nifti1_read_write -r /home/andy/projects/breastAxialSeg/tuneSagModel/RIA_19-093_000_01190_20161031/T1_axial_slope2.nii /home/andy/projects/breastAxialSeg/tuneSagModel/RIA_19-093_000_01190_20161031/T1_axial_slope2.nii')
tt=time.time()
#hdr = c_char_p('/home/andy/projects/breastAxialSeg/tuneSagModel/RIA_19-093_000_01190_20161031/T1_axial_02_01.nii')
#img = c_char_p('/home/andy/projects/breastAxialSeg/tuneSagModel/RIA_19-093_000_01190_20161031/T1_axial_02_01.nii')
img = c_char_p('/home/andy/projects/breastAxialSeg/tuneSagModel/RIA_19-093_000_06814_20141024/T1_axial_02_01.nii')
###x = c_float()
###x = Array(c_float * 54525952)
###x = (c_float * 54525952)(0)
###x = np.zeros((208,512,512),dtype=c_float)
###p = pointer(c_float)
###fun.main(hdr,img,byref(x))
###byref(x) = fun.main(hdr,img)
#x = fun.main(img)
#xx = np.ctypeslib.as_array(x,shape=(54525952,1))
#I = xx.reshape((208,512,512),order='F')
#print('reading took {} s'.format(round(time.time() - tt,2)))
#print(np.sum(xx,dtype='double'))
#plt.imshow(I[85,:,:],cmap='gray'), plt.clim(0,1)

##low = np.array([24,300,151],dtype=c_int)
##high = np.array([149,414,244],dtype=c_int)
##numOfVoxels = (high-low+1).prod()
#low = (c_int*3)(24,300,151)
#high = (c_int*3)(149,414,244)
##x = fun.main(img,byref(low),byref(high),1362060)
#x = fun.main(img,low,high,1362060)
#xx = np.ctypeslib.as_array(x,shape=(1362060,1))
#I = xx.reshape((126,115,94),order='F')
#print('reading took {} s'.format(time.time() - tt,2))
#print(np.sum(xx,dtype='double'))
#plt.imshow(I[63,:,:],cmap='gray'), plt.clim(0,1)

#low = (c_int*3)(3,4,5)
#high = (c_int*3)(5,6,7)
#x = fun.main(img,byref(low),byref(high),27)
#xx = np.ctypeslib.as_array(x,shape=(27,1))
#I = xx.reshape((3,3,3),order='F')
#print('reading took {} s'.format(time.time() - tt,2))
#print(np.sum(xx,dtype='double'))
#plt.imshow(I[1,:,:],cmap='gray'), plt.clim(0,1)

#low = (c_int*3)(93,365,175)
#high = (c_int*3)(95,366,175)
#x = fun.main(img,byref(low),byref(high),6)
#xx = np.ctypeslib.as_array(x,shape=(6,1))
#I = xx.reshape((3,2,1),order='F')
#print('reading took {} s'.format(time.time() - tt,2))
#print(np.sum(xx,dtype='double'))
##plt.imshow(I[1,:,:],cmap='gray'), plt.clim(0,1)

#img = c_char_p('/home/andy/projects/breastAxialSeg/tuneSagModel/RIA_19-093_000_06814_20141024/T1_axial_02_01.nii')
#
low = (c_int*3)(83, 180, 302)
high = (c_int*3)(101, 254, 376)
#low = (c_int*3)(93, 180, 302)
#high = (c_int*3)(112, 255, 377)
#x = fun.main(img,byref(low),byref(high),106875)
x = fun.main(img,low,high,106875)
xx = np.ctypeslib.as_array(x,shape=(106875,1))
I = xx.reshape((19,75,75),order='F')
print('reading took {} s'.format(time.time() - tt,2))
print(np.sum(xx,dtype='double'))
plt.imshow(I[10,:,:],cmap='gray'), plt.clim(0,1)
