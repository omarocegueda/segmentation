import numpy as np
import scipy as sp
import nibabel as nib
import os
import argparse
#import math
import time
import matplotlib.pyplot as plt
#from scipy import signal
from PIL import Image
from pylab import *
#from sklearn.preprocessing import normalize
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces.ants import Registration
from dipy.align.reslice import reslice
from skimage.restoration import denoise_bilateral

## Applies preprocessing steps to the T1 and T2 subject NIFTY images.
## Input   :  the name of the nifty T1 subject file t1FileName
##            and the name of the nifty T2 subject file t2FileName.
## Output  :  T1 and T2 preprocessed NIFTY images (in numpy arrays).
def preprocessingSteps(t1FileName,t2FileName):
    
    # Step 1.1 - Intensity inhomogeneity correction
    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = t1FileName
    t1FileName = t1FileName[0:(t1FileName.index(".nii"))] + "_1-1.nii.gz"
    n4.inputs.output_image = t1FileName
    n4.cmdline
    n4.run()
    n4_2 = N4BiasFieldCorrection()
    n4_2.inputs.dimension = 3
    n4_2.inputs.input_image = t2FileName
    t2FileName = t2FileName[0:(t2FileName.index(".nii"))] + "_1-1.nii.gz"
    n4_2.inputs.output_image = t2FileName
    n4_2.cmdline
    n4_2.run()
    
    # Step 1.2 - Rigid Registration of T2 to T1
    reg = Registration()
    reg.inputs.fixed_image = t1FileName
    reg.inputs.moving_image= t2FileName
    reg.inputs.metric = ['Mattes'] #['Demons']
    reg.inputs.metric_weight = [1]
    reg.inputs.shrink_factors = [[2]]
    reg.inputs.smoothing_sigmas = [[1]]
    reg.inputs.transforms = ['Rigid']
    reg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
    reg.inputs.dimension = 3
    reg.inputs.number_of_iterations = [[1500]]
    reg.inputs.convergence_threshold = [1.e-8]
    t2FileName = t2FileName[0:(t2FileName.index("_1-1.nii"))] + "_1-2.nii.gz"
    reg.inputs.output_warped_image = t2FileName
    reg.cmdline
    reg.run()
    
    # Step 1.3 - Alignment to neurological orientation
    imgSubjectT1 = nib.load(t1FileName)
    imgSubjectT2 = nib.load(t2FileName)
    axCodesT1 = nib.aff2axcodes(imgSubjectT1.affine)
    axCodesT2 = nib.aff2axcodes(imgSubjectT2.affine)
    if not (axCodesT1[0]=='R' and axCodesT1[1]=='A' and axCodesT1[2]=='S') :
       # reorient
       imgSubjectT1 = nib.funcs.as_closest_canonical(imgSubjectT1)
    if not (axCodesT2[0]=='R' and axCodesT2[1]=='A' and axCodesT2[2]=='S') :
       # reorient
       imgSubjectT2 = nib.funcs.as_closest_canonical(imgSubjectT2)
    
    imgSubjectData_T1 = np.array(imgSubjectT1.get_data())
    imgSubjectData_T2 = np.array(imgSubjectT2.get_data())
    
    dimX = imgSubjectData_T1.shape[0]
    dimY = imgSubjectData_T1.shape[1]
    dimZ = imgSubjectData_T1.shape[2]
    # Step 1.4 - Resampling for isotropic voxels
    affineT1  = imgSubjectT1.get_affine()
    zoomsT1   = imgSubjectT1.get_header().get_zooms()[:3]
    n_zoomsT1 = (0.6,0.6,0.6) #(1.,1.,1.)
    imgSubjectData_T1,affineT1 = reslice(imgSubjectData_T1,affineT1,zoomsT1,n_zoomsT1)
    affineT2  = imgSubjectT2.get_affine()
    zoomsT2   = imgSubjectT2.get_header().get_zooms()[:3]
    n_zoomsT2 = (0.6,0.6,0.6) #(1.,1.,1.)
    imgSubjectData_T2,affineT2 = reslice(imgSubjectData_T2,affineT2,zoomsT2,n_zoomsT2)
    # Step 1.5 - Anisotropic diffusion filter
    maxVal = imgSubjectData_T1.max()
    minVal = imgSubjectData_T1.min()
    imgSubjectData_T1 = (1.0/(maxVal-minVal))*(imgSubjectData_T1-minVal)
    maxVal = imgSubjectData_T2.max()
    minVal = imgSubjectData_T2.min()
    imgSubjectData_T2 = (1.0/(maxVal-minVal))*(imgSubjectData_T2-minVal)
    imgSubjectData_T1 = denoise_bilateral(imgSubjectData_T1)
    imgSubjectData_T2 = denoise_bilateral(imgSubjectData_T2)
    
    # show a sample resulting slice
    nSlice = 70
    ImT1 = Image.fromarray(np.uint8(cm.Greys_r(imgSubjectData_T1[:,:,nSlice])*255.0))
    ImT1.show()
    ImT2 = Image.fromarray(np.uint8(cm.Greys_r(imgSubjectData_T2[:,:,nSlice])*255.0))
    ImT2.show()
    # save images
    ST1 = nib.Nifti1Image(imgSubjectData_T1,affineT1)
    ST2 = nib.Nifti1Image(imgSubjectData_T2,affineT2)
    print "Preprocessed shape: {}".format(ST1.shape)
    t1FileName = t1FileName[0:(t1FileName.index("_1-1.nii"))] + "_1-5.nii.gz"
    t2FileName = t2FileName[0:(t2FileName.index("_1-2.nii"))] + "_1-5.nii.gz"
    nib.save(ST1,t1FileName)
    nib.save(ST2,t2FileName)
    
    
parser = argparse.ArgumentParser(description='Applies preprocessing steps to the subject T1 and T2 image files in NIFTY format.')
parser.add_argument('t1FileName',metavar='t1FileName',type=str,help='name of the T1 subject file')
parser.add_argument('t2FileName',metavar='t2FileName',type=str,help='name of the T2 subject file')

args = parser.parse_args()
t1FileName = args.t1FileName
t2FileName = args.t2FileName


start_time = time.time()
preprocessingSteps(t1FileName,t2FileName)
print "Total time: {} seconds.".format(time.time() - start_time)

