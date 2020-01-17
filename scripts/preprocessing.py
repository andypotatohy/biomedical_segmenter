# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:28:09 2019

@author: hirsch
"""
import nibabel as nib
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt


def rescale(img, min_new = 0, max_new = 255):
    " Rescale image. Default is int8 [0-255] "
    return ((img - img.min()) * (float(max_new - min_new) / float(img.max() - img.min()))) + min_new
    

def percentile95_normalizeMRI(data):
    p95 = np.percentile(data,95)
    data1 = data/p95
    return(data1)

nii = nib.load('/home/hirsch/Documents/projects/stroke_heads/MODEL_INPUT/MRI/original/rNC019_NC019_Background1_Background 1_COPY.nii')
d = nii.get_data()
print(nii.header['pixdim'][1:4])
res = nii.header['pixdim'][1:4]
shape = d.shape
target_res = [3, 0.75, 0.75]

out_shape = np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])
img2 = resize(d, output_shape=out_shape, preserve_range=True, anti_aliasing=True)



#img2_out = nib.Nifti1Image(img2, np.diag((target_res + [0])))
#nib.save(img2_out, '/home/hirsch/Documents/projects/stroke_heads/MODEL_INPUT/MRI/original/python_rrNC019_NC019_Background1_Background 1_COPY.nii')


nii2 = nib.load('/home/hirsch/Documents/projects/stroke_heads/MODEL_INPUT/MRI/original/rrNC019_NC019_Background1_Background 1_COPY.nii')

img_ori = nii2.get_data()
img_ori.shape

print(nii2.header)

img2.shape


img3 = resize(d, output_shape=img_ori.shape, preserve_range=True)

np.max(img2)

sub = img_ori[20,:,:] - img2[20,:,:]
np.max(sub)
np.mean(sub)
np.std(sub)
plt.imshow(sub )






nii = nib.load('/media/hirsch/Ivan Backup/alignedNii/MSKCC_16-328_1_11973_20080422/t1post-r.nii')
d = nii.get_data()

res = nii.header['pixdim'][1:4]
shape = d.shape
target_res = [3, 0.75, 0.75]
out_shape = np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])
img2 = resize(d, output_shape=out_shape, preserve_range=True, anti_aliasing=True)

img2 = rescale(img2, min_new=0, max_new=255)
img2 = percentile95_normalizeMRI(img2)



img2_out = nib.Nifti1Image(img2, np.diag((target_res + [0])))
nib.save(img2_out,'/media/hirsch/Ivan Backup/test_img.nii')