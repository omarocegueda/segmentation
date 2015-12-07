import numpy as np
import scipy as sp
import nibabel as nib
import os
import argparse
import time
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *
from skimage.morphology import ball,opening,dilation,erosion,watershed
from skimage.filters import threshold_otsu

## Applies Intracraneal Cavity Extraction to the T1 and T2 subject NIFTY images.
## Input   :  the name of the preprocessed nifty T1 subject file t1FileName
##            and the name of the nifty preprocessed T2 subject file t2FileName.
## Output  :  T1 and T2 preprocessed NIFTY images (in numpy arrays).
def ICE(t1FileName,t2FileName):
    # load images
    imgSubjectT1 = nib.load(t1FileName)
    imgSubjectT2 = nib.load(t2FileName)    
    imgSubjectData_T1 = np.array(imgSubjectT1.get_data())
    imgSubjectData_T2 = np.array(imgSubjectT2.get_data())
    
    dimX = imgSubjectData_T1.shape[0]
    dimY = imgSubjectData_T1.shape[1]
    dimZ = imgSubjectData_T1.shape[2]
    # define structuring element
    struct_elem = ball(1)
    # Perform morphologic opening on T2 image
    openedT2 = opening(imgSubjectData_T2,struct_elem)
    # Obtain morphological gradient of opened T2 image
    dilationOT2 = dilation(openedT2,struct_elem)
    erosionOT2  = erosion(openedT2,struct_elem)
    gradientOT2 = dilationOT2 - erosionOT2
    # Obtain segmentation function (sum of increasing scale dilations)
    dilGradOT2_1 = dilation(gradientOT2,ball(1))
    dilGradOT2_2 = dilation(gradientOT2,ball(2))
    dilGradOT2_3 = dilation(gradientOT2,ball(3))
    segFuncGOT2  = dilGradOT2_1 + dilGradOT2_2 + dilGradOT2_3
    # Obtain T2 mask by threshold
    t = threshold_otsu(imgSubjectData_T2)
    maskT2 = imgSubjectData_T2 >= (t*2.1)
    maskT2 = np.array(maskT2,dtype=float)
    # Obtain gravity center of mask of T2
    C = np.zeros(3)
    maskT2Count = 0
    for x in xrange(0,dimX):
      for y in xrange(0,dimY):
        for z in xrange(0,dimZ):
           if maskT2[x,y,z] > 0 :
              maskT2Count = maskT2Count + 1
              C[0] = C[0] + x
              C[1] = C[1] + y
              C[2] = C[2] + z
    C = C / float(maskT2Count)
    print "Shape : {}".format(maskT2.shape)
    print "Centroid = {}".format(C)
    # set markers
    markersICE = np.array(np.zeros((dimX,dimY,dimZ)),dtype=int)
    markersICE[int(C[0]),int(C[1]),int(C[2])] = 2
    markersICE[int(C[0])+1,int(C[1]),int(C[2])] = 2
    markersICE[int(C[0])-1,int(C[1]),int(C[2])] = 2
    markersICE[int(C[0]),int(C[1])+1,int(C[2])] = 2
    markersICE[int(C[0]),int(C[1])-1,int(C[2])] = 2
    markersICE[int(C[0]),int(C[1]),int(C[2])+1] = 2
    markersICE[int(C[0]),int(C[1]),int(C[2])-1] = 2
    for y in xrange(0,dimY):
      for z in xrange(0,dimZ):
         markersICE[0,y,z] = 1
         markersICE[dimX-1,y,z] = 1
    for x in xrange(0,dimX):
      for z in xrange(0,dimZ):
         markersICE[x,0,z] = 1
         markersICE[x,dimY-1,z] = 1
    for y in xrange(0,dimY):
      for x in xrange(0,dimX):
         markersICE[x,y,0] = 1
         markersICE[x,y,dimZ-1] = 1
    # Apply watershed segmentation with markers
    segFuncGOT2_int = np.array(255.0*segFuncGOT2,dtype=int)
    ICEMask = watershed(segFuncGOT2_int,markersICE)
    # Apply Inctracraneal Cavity Extraction with segmented watershed mask
    for x in xrange(0,dimX):
      for y in xrange(0,dimY):
        for z in xrange(0,dimZ):
           if ICEMask[x,y,z] == 1 :
              imgSubjectData_T1[x,y,z] = 0
              imgSubjectData_T2[x,y,z] = 0
    # show a sample resulting slice
    nSlice = 70
    ImICEMask = Image.fromarray(np.uint8(ICEMask[:,:,nSlice]*127))
    ImICEMask.show()
    ImT1 = Image.fromarray(np.uint8(cm.Greys_r(imgSubjectData_T1[:,:,nSlice])*255.0))
    ImT1.show()
    ImT2 = Image.fromarray(np.uint8(cm.Greys_r(imgSubjectData_T2[:,:,nSlice])*255.0))
    ImT2.show()
    # save images
    #ST1 = nib.Nifti1Image(imgSubjectData_T1,affineT1)
    #ST2 = nib.Nifti1Image(imgSubjectData_T2,affineT2)
    #t1FileName = t1FileName[0:(t1FileName.index("_1-1.nii"))] + "_1-5.nii.gz"
    #t2FileName = t2FileName[0:(t2FileName.index("_1-2.nii"))] + "_1-5.nii.gz"
    #nib.save(ST1,t1FileName)
    #nib.save(ST2,t2FileName)
    
    
parser = argparse.ArgumentParser(description='Applies Intracraneal Cavity Extraction and Hemisphere Separation to the subject T1 and T2 image files in NIFTY format.')
parser.add_argument('t1FileName',metavar='t1FileName',type=str,help='name of the T1 subject file')
parser.add_argument('t2FileName',metavar='t2FileName',type=str,help='name of the T2 subject file')

args = parser.parse_args()
t1FileName = args.t1FileName
t2FileName = args.t2FileName


start_time = time.time()
ICE(t1FileName,t2FileName)
print "Total time: {} seconds.".format(time.time() - start_time)

