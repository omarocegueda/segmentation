import numpy as np
import scipy as sp
import math
import nibabel as nib
import os
import time
import matplotlib.pyplot as plt
import operator
from PIL import Image
from pylab import *
from skimage.morphology import ball,opening,closing,dilation,erosion,watershed,skeletonize
from skimage.restoration import denoise_bilateral
from dipy.align.reslice import reslice
from scipy import ndimage
from fast_morph import (SequencialSphereDilation,
                                  create_sphere,
                                  get_list,
                                  get_subsphere_lists,
                                  isotropic_erosion,
                                  isotropic_dilation)

# Normalize intensity values to the range [0,scaleVal]
# Input:  data numpy array with the intensity values to normalize and
#         scaleVal value for the normalization.
# Output: data numpy array with the normalized intensity values.
def NormalizeIntensity(data,scaleVal):
   maxVal = data.max()
   minVal = data.min()
   data = (scaleVal/(maxVal-minVal))*(data-minVal)
   return data

# Evaluation of Gaussian function.
# Input:  x observation to evaluate, mu mean of the Gaussian and sigma
#         standard deviation of the Gaussian.
# Output: scalar with evaluation of the Gaussian.
def Gaussian(x,mu,sigma):
   result = math.exp( -0.5*((x-mu)**2) / (sigma**2) ) / (math.sqrt(2*math.pi)*sigma)
   return result

# Vectorized version of the Gaussian function.
GaussianVect = np.vectorize(Gaussian, otypes=[np.float])

# Evaluation of Markov Random Field Potential (MRF) function.
# First set parameters based on (Makropoulos, 2015)
b = 1.0
c = 5.0
# Input:  k current class to evaluate (0 - CSF, 1 - UWM, 2 - CGM, 3 - SGM),
#         K total number of classes, p numpy array with probabilities of
#         eack class 1 ... K for each voxel, voxel position variables
#         {ix,iy,iz} and anisotropic voxel spacing variables {sx,sy,sz} =
#         {1/dx, 1/dy, 1/dz} with d the physical spacing in milimiters for
#         each respective direction between two voxels.
# Output: scalar with evaluation of MRF function.
def UMRF(k,K,p,ix,iy,iz,sx,sy,sz):
   totalSum = 0
   innerSum = 0
   for j in range(0,K):
     if k <> j :
       if (k == 2 or k == 3) and (j == 2 or j == 3) : # CGM - SGM (are not neighbouring structures)
          A = c
       else : # all others are neighbouring structures
          A = b
       innerSum = sx*p[ix-1,iy,iz,j] + sx*p[ix+1,iy,iz,j]
       innerSum = innerSum + sy*p[ix,iy-1,iz,j] + sy*p[ix,iy+1,iz,j]
       innerSum = innerSum + sz*p[ix,iy,iz-1,j] + sz*p[ix,iy,iz+1,j]
       totalSum = totalSum + A*innerSum
   return totalSum


# read directories and atlas label
base_dir    = ' '
neo_subject = ' '
results_dir = ' '
atlas_label = ' '
middir      = ' '
with open('directories_and_labels2.txt') as fp :
   i = 0
   for line in fp :
      if i == 0 :
         base_dir = line[0:(len(line)-1)]
      elif i == 1 :
         neo_subject = line[0:(len(line)-1)]
      elif i == 2 :
         results_dir = line[0:(len(line)-1)]
      elif i == 3 :
            atlas_label = line[0:(len(line)-1)]
      else :
         if i == 4 :
            middir = line[0:(len(line)-1)]
      i = i + 1


# Read subject files
t2CurrentSubjectName  = base_dir + middir +neo_subject+'T2_1-1.nii.gz'
t1CurrentSubjectName  = base_dir + middir +neo_subject+'T1_1-2.nii.gz'
GTName                = base_dir + middir +neo_subject+'manualSegm.nii.gz'
PrelSegmName          = base_dir + middir +neo_subject+'SegMapVolume.nii.gz'
#PrelSegmName          = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'SegMapVolumeClosings.nii.gz'
t2CurrentSubject_data = nib.load(t2CurrentSubjectName).get_data()
t1CurrentSubject_data = nib.load(t1CurrentSubjectName).get_data()
GT_data               = nib.load(GTName).get_data()
PrelSegm_data         = nib.load(PrelSegmName).get_data()
affineT2CS            = nib.load(t2CurrentSubjectName).get_affine()
zoomsT2CS             = nib.load(t2CurrentSubjectName).get_header().get_zooms()[:3]


# Read priors files
#AT1Name    = base_dir + middir +neo_subject+'A'+atlas_label+'_T1.nii.gz'
#AT2Name    = base_dir + middir +neo_subject+'A'+atlas_label+'_T2.nii.gz'
AMaskName  = base_dir + middir +neo_subject+'A'+atlas_label+'_Mask.nii.gz'
#ABSName    = base_dir + middir +neo_subject+'A'+atlas_label+'_BS.nii.gz'
#ACeName    = base_dir + middir +neo_subject+'A'+atlas_label+'_Ce.nii.gz'
ACoName    = base_dir + middir +neo_subject+'A'+atlas_label+'_Co.nii.gz'
ACSFName   = base_dir + middir +neo_subject+'A'+atlas_label+'_CSF.nii.gz'
ADGMName   = base_dir + middir +neo_subject+'A'+atlas_label+'_DGM.nii.gz'
AWMName    = base_dir + middir +neo_subject+'A'+atlas_label+'_WM.nii.gz'
A50Name    = base_dir + middir +neo_subject+'A'+atlas_label+'_50.nii.gz'
#AT1_data   = nib.load(AT1Name).get_data()
#AT2_data   = nib.load(AT2Name).get_data()
AMask_data = nib.load(AMaskName).get_data()
#ABS_data   = nib.load(ABSName).get_data()
#ACe_data   = nib.load(ACeName).get_data()
ACo_data   = nib.load(ACoName).get_data()
ACSF_data  = nib.load(ACSFName).get_data()
ADGM_data  = nib.load(ADGMName).get_data()
AWM_data   = nib.load(AWMName).get_data()
A50_data   = nib.load(A50Name).get_data()


start_time = time.time()

# Step 1.4 - Resampling for isotropic voxels

n_zooms = (zoomsT2CS[0],zoomsT2CS[0],zoomsT2CS[0])
v = n_zooms[0]
t2CurrentSubject_data,affineT2CS = reslice(t2CurrentSubject_data,affineT2CS,zoomsT2CS,n_zooms)
t1CurrentSubject_data,_          = reslice(t1CurrentSubject_data,affineT2CS,zoomsT2CS,n_zooms)
#AT1_data,_                       = reslice(AT1_data,affineT2CS,zoomsT2CS,n_zooms)
#AT2_data,_                       = reslice(AT2_data,affineT2CS,zoomsT2CS,n_zooms)
AMask_data,_                     = reslice(AMask_data,affineT2CS,zoomsT2CS,n_zooms)
#ABS_data,_                       = reslice(ABS_data,affineT2CS,zoomsT2CS,n_zooms)
#ACe_data,_                       = reslice(ACe_data,affineT2CS,zoomsT2CS,n_zooms)
ACo_data,_                       = reslice(ACo_data,affineT2CS,zoomsT2CS,n_zooms)
ACSF_data,_                      = reslice(ACSF_data,affineT2CS,zoomsT2CS,n_zooms)
ADGM_data,_                      = reslice(ADGM_data,affineT2CS,zoomsT2CS,n_zooms)
AWM_data,_                       = reslice(AWM_data,affineT2CS,zoomsT2CS,n_zooms)
A50_data,_                       = reslice(A50_data,affineT2CS,zoomsT2CS,n_zooms,order=0)

# Step 1.5 - Anisotropic diffusion filter

scaleValue = 1.0
t2CurrentSubject_data = denoise_bilateral(NormalizeIntensity(t2CurrentSubject_data,scaleValue),win_size=5)
t1CurrentSubject_data = denoise_bilateral(NormalizeIntensity(t1CurrentSubject_data,scaleValue),win_size=5)


# Normalize the rest of the volume intensity values to [0,255]
scaleValue            = 255.0
t2CurrentSubject_data = NormalizeIntensity(t2CurrentSubject_data,scaleValue)
t1CurrentSubject_data = NormalizeIntensity(t1CurrentSubject_data,scaleValue)
#AT1_data              = NormalizeIntensity(AT1_data,scaleValue)
#AT2_data              = NormalizeIntensity(AT2_data,scaleValue)
AMask_data            = NormalizeIntensity(AMask_data,scaleValue)
#ABS_data              = NormalizeIntensity(ABS_data,scaleValue)
#ACe_data              = NormalizeIntensity(ACe_data,scaleValue)
ACo_data              = NormalizeIntensity(ACo_data,scaleValue)
ACSF_data             = NormalizeIntensity(ACSF_data,scaleValue)
ADGM_data             = NormalizeIntensity(ADGM_data,scaleValue)
AWM_data              = NormalizeIntensity(AWM_data,scaleValue)

dim1 = t2CurrentSubject_data.shape[0]
dim2 = t2CurrentSubject_data.shape[1]
dim3 = t2CurrentSubject_data.shape[2]


# 2 - Intracranial Cavity Extraction

#   apply atlas head mask
t2CurrentSubject_data[ AMask_data == 0 ] = 0


nSlice = int(dim3 / 2)
difSlice = int(dim3/10)
nSliceGT = int(GT_data.shape[2] / 2)
difSliceGT = int(GT_data.shape[2]/10)


#   define structuring element
struct_elem = ball(5) # <-- should have 9 voxel units of diameter
#   Perform morphologic opening on T2 image
openedT2 = opening(t2CurrentSubject_data,struct_elem)
Im = Image.fromarray(np.uint8(openedT2[:,:,nSlice]))
Im.save(results_dir+'openedT2.png')

#   Obtain morphological gradient of opened T2 image
cross_se = ball(1) # <-- should have 3 voxel units of diameter
dilationOT2 = dilation(openedT2,cross_se)
erosionOT2  = erosion(openedT2,cross_se)
del openedT2
gradientOT2 = dilationOT2 - erosionOT2
Im = Image.fromarray(np.uint8(gradientOT2[:,:,nSlice]))
Im.save(results_dir+'GradientMorp_NoNorm.png')
del dilationOT2
del erosionOT2
gradientOT2 = NormalizeIntensity(gradientOT2,255.0)
Im = Image.fromarray(np.uint8(gradientOT2[:,:,nSlice]))
Im.save(results_dir+'GradientMorp_Norm.png')

#   Obtain segmentation function (sum of increasing scale dilations)
SSD             = SequencialSphereDilation(gradientOT2)
nScaleDilations = 5 # counts dilation at 0 radius
dilGradOT2      = gradientOT2
for r in range(1,nScaleDilations):
   SSD.expand(gradientOT2)
   dilGradOT2 = SSD.get_current_dilation() + dilGradOT2

del SSD
segFuncGOT2  = NormalizeIntensity(dilGradOT2,255.0)
Im = Image.fromarray(np.uint8(segFuncGOT2[:,:,nSlice]))
Im.save(results_dir+'seg_func_ICE.png')
del dilGradOT2
#del gradientOT2


#   Obtain gravity center of mask of T2
C    = np.zeros(3)
CenM = ndimage.measurements.center_of_mass(t2CurrentSubject_data)
C[0] = CenM[0]
C[1] = CenM[1]
C[2] = CenM[2]

print "Centroid = {}".format(C)

#   set two class of markers (for marker based watershed segmentation)
markersICE = np.array(np.zeros((dim1,dim2,dim3)),dtype=int)
markersICE[int(C[0]),int(C[1]),int(C[2])] = 2
for i in range(1,4):
  markersICE[int(C[0])+i,int(C[1]),int(C[2])] = 2
  markersICE[int(C[0])-i,int(C[1]),int(C[2])] = 2
  markersICE[int(C[0]),int(C[1])+i,int(C[2])] = 2
  markersICE[int(C[0]),int(C[1])-i,int(C[2])] = 2
  markersICE[int(C[0]),int(C[1]),int(C[2])+i] = 2
  markersICE[int(C[0]),int(C[1]),int(C[2])-i] = 2

for y in range(0,dim2):
  for z in range(0,dim3):
     markersICE[0,y,z] = 1
     markersICE[dim1-1,y,z] = 1

for x in range(0,dim1):
  for z in range(0,dim3):
     markersICE[x,0,z] = 1
     markersICE[x,dim2-1,z] = 1

for y in range(0,dim2):
  for x in range(0,dim1):
     markersICE[x,y,0] = 1
     markersICE[x,y,dim3-1] = 1

#   Apply watershed segmentation with markers
segFuncGOT2 = np.array(segFuncGOT2,dtype=int)
ICEMask = watershed(segFuncGOT2,markersICE)
del segFuncGOT2
del markersICE
ICEMask = dilation(ICEMask,ball(1))
#   Apply Inctracranial Cavity Extraction with segmented watershed mask
t2CurrentSubject_data[ ICEMask == 1 ] = 0
t1CurrentSubject_data[ ICEMask == 1 ] = 0


#   show a sample resulting slice

Im = Image.fromarray(np.uint8(ICEMask[:,:,nSlice]*127))
Im.save(results_dir+'ICEMask.png')
Im = Image.fromarray(np.uint8(ICEMask[:,:,nSlice+difSlice]*127))
Im.save(results_dir+'ICEMask_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(ICEMask[:,:,nSlice-difSlice]*127))
Im.save(results_dir+'ICEMask_m'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(t1CurrentSubject_data[:,:,nSlice]))
Im.save(results_dir+'t1CS.png')
Im = Image.fromarray(np.uint8(t1CurrentSubject_data[:,:,nSlice+difSlice]))
Im.save(results_dir+'t1CS_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(t1CurrentSubject_data[:,:,nSlice-difSlice]))
Im.save(results_dir+'t1CS_m'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(t2CurrentSubject_data[:,:,nSlice]))
Im.save(results_dir+'t2CS.png')
Im = Image.fromarray(np.uint8(t2CurrentSubject_data[:,:,nSlice+difSlice]))
Im.save(results_dir+'t2CS_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(t2CurrentSubject_data[:,:,nSlice-difSlice]))
Im.save(results_dir+'t2CS_m'+str(difSlice)+'.png')


#   Get bounding box coordinates to reduce computations

maxI = 0
maxJ = 0
maxK = 0
minI = dim1
minJ = dim2
minK = dim3
for i in range(0,dim1):
  for j in range(0,dim2):
   for k in range(0,dim3):
      if ICEMask[i,j,k] == 2 :
         if i < minI :
            minI = i
         if j < minJ :
            minJ = j
         if k < minK :
            minK = k
         if i > maxI :
            maxI = i
         if j > maxJ :
            maxJ = j
         if k > maxK :
            maxK = k

print "bounding box i:(min={},max={}), j:(min={},max={}), k:(min={},max={}).".format(minI,maxI,minJ,maxJ,minK,maxK)

print "Until ICE: {} seconds.".format(time.time() - start_time)


del ICEMask


#   Remove unconnected CSF inside UWM
UWMBinary = np.array(np.ones((dim1,dim2,dim3)),dtype=int)
UWMBinary[ PrelSegm_data == 190 ] = 0
UWMBinary,numcomponents = sp.ndimage.measurements.label(UWMBinary)

componentLabels = []
for i_nc in range(0,numcomponents):
   labelId = i_nc + 1
   componentLabels.append( (sum(UWMBinary==labelId),labelId) )

componentLabels.sort(key=lambda tup: tup[0],reverse=True)
#     keep only Largest Connected Component
for i in range(minI,maxI+1):
  for j in range(minJ,maxJ+1):
    for k in range(minK,maxK+1):
       if UWMBinary[i,j,k] <> componentLabels[0][1] :
          PrelSegm_data[i,j,k] = 190


print "UWM Corrected"
del UWMBinary
Im = Image.fromarray(np.uint8(PrelSegm_data[:,:,nSlice]))
Im.save(results_dir+'PrelSegmUWMCorrected.png')
Im = Image.fromarray(np.uint8(PrelSegm_data[:,:,nSlice+difSlice]))
Im.save(results_dir+'PrelSegmUWMCorrected_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(PrelSegm_data[:,:,nSlice-difSlice]))
Im.save(results_dir+'PrelSegmUWMCorrected_m'+str(difSlice)+'.png')



#   Generate Cortical GM prior information

#     Obtain skeleton by coronal slice

GMSkeleton = np.array(np.zeros((dim1,dim2,dim3)),dtype=int)
GMSkeleton[ PrelSegm_data == 85 ] = 1
for k in range(minK,maxK+1):
   GMSkeleton[:,:,k] = skeletonize(GMSkeleton[:,:,k])



filterTol = 1.0e-08
xTol      = - math.log(filterTol)
#     Obtain greatest Hessian norm in the cortical regions
grads  = np.gradient(t1CurrentSubject_data)
Ix     = grads[0]
Iy     = grads[1]
Iz     = grads[2]
del grads
gradsx = np.gradient(Ix)
Ixx    = gradsx[0]
Ixy    = gradsx[1]
Ixz    = gradsx[2]
del gradsx
gradsy = np.gradient(Iy)
Iyx    = gradsy[0]
Iyy    = gradsy[1]
Iyz    = gradsy[2]
del gradsy
gradsz = np.gradient(Iz)
Izx    = gradsz[0]
Izy    = gradsz[1]
Izz    = gradsz[2]
del gradsz

maxHessNorm = 0.0
for i in range(minI,maxI+1):
 for j in range(minJ,maxJ+1):
  for k in range(minK,maxK+1):
    val = t2CurrentSubject_data[i,j,k]
    if val >0 : 
          #if ACo_data[i,j,k] >= 51 :
          cond = PrelSegm_data[i,j,k] == 85
          if cond :
             # compute Hessian with central finite differences
             currHessNorm = Ixx[i,j,k]**2 + Ixy[i,j,k]**2 +  Ixz[i,j,k]**2
             currHessNorm = currHessNorm + Iyx[i,j,k]**2 + Iyy[i,j,k]**2 +  Iyz[i,j,k]**2
             currHessNorm = currHessNorm + Izx[i,j,k]**2 + Izy[i,j,k]**2 +  Izz[i,j,k]**2
             currHessNorm = math.sqrt(currHessNorm)
             if currHessNorm > maxHessNorm :
                maxHessNorm = currHessNorm


print "maxHessNorm={}".format(maxHessNorm)
#     Apply line-like and plate-like filters based on eigenvalues
#     of the Hessian matrix on the T1 image
CGMPriorT1 = np.zeros((dim1,dim2,dim3),dtype=float)
c      = 0.5*maxHessNorm
alpha  = 0.5
betha  = 0.5
maxS   = 0

for i in range(minI,maxI+1):
 for j in range(minJ,maxJ+1):
  for k in range(minK,maxK+1):
    val = t2CurrentSubject_data[i,j,k]
    if val >0 : 
          #if ACo_data[i,j,k] >= 51 :
          cond = PrelSegm_data[i,j,k] == 85
          if cond :
             # compute Hessian with central finite differences
             H = np.array([[Ixx[i,j,k], Ixy[i,j,k], Ixz[i,j,k]], [Iyx[i,j,k], Iyy[i,j,k], Iyz[i,j,k]], [Izx[i,j,k], Izy[i,j,k], Izz[i,j,k]]],dtype=float)
             # get ordered eigenvalues
             egval, _ = np.linalg.eig(H)
             egvalList = [ (abs(egval[0]),egval[0]), (abs(egval[1]),egval[1]), (abs(egval[2]),egval[2]) ]
             egvalList.sort(key=lambda tup: tup[0])
             Lamb1 = egvalList[0][1]
             Lamb2 = egvalList[1][1]
             Lamb3 = egvalList[2][1]
             
             # apply line-like filter and plate-like filter to current voxel
             S = math.sqrt(Lamb1**2 + Lamb2**2 + Lamb3**2)
             if S > maxS :
                maxS = S
             Rb = abs(Lamb1) / math.sqrt(abs(Lamb2*Lamb3))
             Ra = abs(Lamb2) / abs(Lamb3)
             Vline = 0
             if Lamb2 < 0 and Lamb3 < 0 :
                Vline = 1.0 - math.exp( -(Ra**2)/(2.0*(alpha**2)) )
                Vline = Vline * math.exp( -(Rb**2)/(2.0*(betha**2)) )
                Vline = Vline * ( 1.0 - math.exp( -(S**2)/(2.0*(c**2)) ) )
             Vplate = 0
             if Lamb3 < 0 :
                Vplate = math.exp( -(Ra**2)/(2.0*(alpha**2)) )
                Vplate = Vplate * math.exp( -(Rb**2)/(2.0*(betha**2)) )
                Vplate = Vplate * ( 1.0 - math.exp( -(S**2)/(2.0*(c**2)) ) )
             CGMPriorT1[i,j,k] = max(filterTol,max(Vline,Vplate))
             if CGMPriorT1[i,j,k] > 0.4 : #0.004 :
                CGMPriorT1[i,j,k] = filterTol
             CGMPriorT1[i,j,k] = math.log(CGMPriorT1[i,j,k]) + xTol


print "maxS={}, maxCGMPriorT1={}, minCGMPriorT1={}".format(maxS,CGMPriorT1.max(),CGMPriorT1.min())
print "histogramT1={}".format(np.histogram(CGMPriorT1,bins=100))
CGMPriorT1 = NormalizeIntensity(CGMPriorT1,255.0)

Im = Image.fromarray(np.uint8(CGMPriorT1[:,:,nSlice]))
Im.save(results_dir+'CGMPriorT1.png')
Im = Image.fromarray(np.uint8(CGMPriorT1[:,:,nSlice+difSlice]))
Im.save(results_dir+'CGMPriorT1_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(CGMPriorT1[:,:,nSlice-difSlice]))
Im.save(results_dir+'CGMPriorT1_m'+str(difSlice)+'.png')

grads  = np.gradient(t2CurrentSubject_data)
Ix     = grads[0]
Iy     = grads[1]
Iz     = grads[2]
del grads
gradsx = np.gradient(Ix)
Ixx    = gradsx[0]
Ixy    = gradsx[1]
Ixz    = gradsx[2]
del gradsx
gradsy = np.gradient(Iy)
Iyx    = gradsy[0]
Iyy    = gradsy[1]
Iyz    = gradsy[2]
del gradsy
gradsz = np.gradient(Iz)
Izx    = gradsz[0]
Izy    = gradsz[1]
Izz    = gradsz[2]
del gradsz

maxHessNorm = 0.0
for i in range(minI,maxI+1):
 for j in range(minJ,maxJ+1):
  for k in range(minK,maxK+1):
    val = t2CurrentSubject_data[i,j,k]
    if val >0 : 
          #if ACo_data[i,j,k] >= 51 :
          cond = PrelSegm_data[i,j,k] == 85
          if cond :
             # compute Hessian with central finite differences
             currHessNorm = Ixx[i,j,k]**2 + Ixy[i,j,k]**2 +  Ixz[i,j,k]**2
             currHessNorm = currHessNorm + Iyx[i,j,k]**2 + Iyy[i,j,k]**2 +  Iyz[i,j,k]**2
             currHessNorm = currHessNorm + Izx[i,j,k]**2 + Izy[i,j,k]**2 +  Izz[i,j,k]**2
             currHessNorm = math.sqrt(currHessNorm)
             if currHessNorm > maxHessNorm :
                maxHessNorm = currHessNorm


print "maxHessNorm={}".format(maxHessNorm)
#     Apply line-like and plate-like filters based on eigenvalues
#     of the Hessian matrix on the T2 image
CGMPrior = np.zeros((dim1,dim2,dim3),dtype=float)
c      = 0.5*maxHessNorm
alpha  = 0.5
betha  = 0.5
maxS   = 0

for i in range(minI,maxI+1):
 for j in range(minJ,maxJ+1):
  for k in range(minK,maxK+1):
    val = t2CurrentSubject_data[i,j,k]
    if val >0 : 
          #if ACo_data[i,j,k] >= 51 :
          cond = PrelSegm_data[i,j,k] == 85
          if cond :
             # compute Hessian with central finite differences
             H = np.array([[Ixx[i,j,k], Ixy[i,j,k], Ixz[i,j,k]], [Iyx[i,j,k], Iyy[i,j,k], Iyz[i,j,k]], [Izx[i,j,k], Izy[i,j,k], Izz[i,j,k]]],dtype=float)
             # get ordered eigenvalues
             egval, _ = np.linalg.eig(H)
             egvalList = [ (abs(egval[0]),egval[0]), (abs(egval[1]),egval[1]), (abs(egval[2]),egval[2]) ]
             egvalList.sort(key=lambda tup: tup[0])
             Lamb1 = egvalList[0][1]
             Lamb2 = egvalList[1][1]
             Lamb3 = egvalList[2][1]
             
             # apply line-like filter and plate-like filter to current voxel
             S = math.sqrt(Lamb1**2 + Lamb2**2 + Lamb3**2)
             if S > maxS :
                maxS = S
             Rb = abs(Lamb1) / math.sqrt(abs(Lamb2*Lamb3))
             Ra = abs(Lamb2) / abs(Lamb3)
             Vline = 0
             if Lamb2 > 0 and Lamb3 > 0 :
                Vline = 1.0 - math.exp( -(Ra**2)/(2.0*(alpha**2)) )
                Vline = Vline * math.exp( -(Rb**2)/(2.0*(betha**2)) )
                Vline = Vline * ( 1.0 - math.exp( -(S**2)/(2.0*(c**2)) ) )
             Vplate = 0
             if Lamb3 > 0 :
                Vplate = math.exp( -(Ra**2)/(2.0*(alpha**2)) )
                Vplate = Vplate * math.exp( -(Rb**2)/(2.0*(betha**2)) )
                Vplate = Vplate * ( 1.0 - math.exp( -(S**2)/(2.0*(c**2)) ) )
             CGMPrior[i,j,k] = max(filterTol,max(Vline,Vplate))
             if CGMPrior[i,j,k] > 0.4 : #0.004 :
                CGMPrior[i,j,k] = filterTol
             CGMPrior[i,j,k] = (255.0/xTol)*(math.log(CGMPrior[i,j,k]) + xTol) + CGMPriorT1[i,j,k] + gradientOT2[i,j,k]
             #CGMPrior[i,j,k] = ( CGMPrior[i,j,k] + 255.0-val + t1CurrentSubject_data[i,j,k] ) / 5.0
             CGMPrior[i,j,k] = ( CGMPrior[i,j,k] ) / 3.0


print "maxS={}, maxCGMPriorT2={}, minCGMPriorT2={}".format(maxS,CGMPrior.max(),CGMPrior.min())
print "histogramT2={}".format(np.histogram(CGMPrior,bins=100))
#CGMPrior = NormalizeIntensity(CGMPrior,255.0)
#    combine the prior CGM information from both T1 and T2 images
#CGMPrior = np.maximum(CGMPriorT1,CGMPrior)
#    average the Hessian filter from T1 and T2 images and the morphological gradient
#CGMPrior = (CGMPriorT1 + CGMPrior + gradientOT2) / 3.0
Im = Image.fromarray(np.uint8(CGMPrior[:,:,nSlice]))
Im.save(results_dir+'CGMPrior.png')
Im = Image.fromarray(np.uint8(CGMPrior[:,:,nSlice+difSlice]))
Im.save(results_dir+'CGMPrior_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(CGMPrior[:,:,nSlice-difSlice]))
Im.save(results_dir+'CGMPrior_m'+str(difSlice)+'.png')

#     Keep as CGM only the AND operation between CGMPrior and PrelSegm_data
#     filling the remaining voxels with CSF.
#ncuboids = 2
#s_2 = 5
#ind = [[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1], [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]]
for i in range(minI,maxI+1):
 for j in range(minJ,maxJ+1):
  for k in range(minK,maxK+1):
    val = t2CurrentSubject_data[i,j,k]
    if val > 0 : #and ACo_data[i,j,k] >= 100 :
          if PrelSegm_data[i,j,k] == 85 and CGMPrior[i,j,k] < 51 : #CGMPrior[i,j,k] == 0 :
             if GMSkeleton[i,j,k] == 0 :
                PrelSegm_data[i,j,k] = 1 #255
             else :
                count = 0
                for i1 in range(-2,3,1):
                 for j1 in range(-2,3,1):
                    if PrelSegm_data[i+i1,j+j1,k] == 190 :
                       count = count + 1
                if count == 0 :
                   PrelSegm_data[i,j,k] = 1 #255
             #for i1 in range(-2,3,1):
             # for j1 in range(-2,3,1):
             #  for k1 in range(-2,3,1):
             #     if PrelSegm_data[i+i1,j+j1,k+k1] == 85 :
             #        count = count + 1
             #if count >= 12 :
             #   PrelSegm_data[i,j,k] = 255


del CGMPriorT1
del CGMPrior


#   Correct partial volume errors
UWMCSFBinary = np.array(np.zeros((dim1,dim2,dim3)),dtype=int)
UWMCSFBinary[ (PrelSegm_data == 190) | (PrelSegm_data==1) ] = 1


for k in range(minK,maxK+1):
   print "Slice {}".format(k)
   UWMCSFBinComp, numcomponents = sp.ndimage.measurements.label(UWMCSFBinary[:,:,k])
   componentLabels = []
   for i_nc in range(0,numcomponents):
      labelId = i_nc + 1
      componentLabels.append( (sum(UWMCSFBinComp==labelId),labelId) )
   componentLabels.sort(key=lambda tup: tup[0],reverse=True)
   if numcomponents > 1 and componentLabels[1][0] >= 0.5*componentLabels[0][0] :
    for i in range(minI,maxI+1):
     for j in range(minJ,maxJ+1):
       if PrelSegm_data[i,j,k] == 1 :
         if UWMCSFBinComp[i,j] == componentLabels[1][1] or UWMCSFBinComp[i,j] == componentLabels[0][1] :
            PrelSegm_data[i,j,k] = 190
         else :
            PrelSegm_data[i,j,k] = 255
   else :
    for i in range(minI,maxI+1):
     for j in range(minJ,maxJ+1):
       if PrelSegm_data[i,j,k] == 1 :
         if UWMCSFBinComp[i,j] == componentLabels[0][1] :
            PrelSegm_data[i,j,k] = 190
         else :
            PrelSegm_data[i,j,k] = 255


del UWMCSFBinary


#   Filter undesirable CSF labeled inside UWM near border
#   regions with Cortical GM.

for rep in range(0,30):
 changesCount = 0
 for i in range(minI,maxI+1):
  for j in range(minJ,maxJ+1):
   for k in range(minK,maxK+1):
     val = t2CurrentSubject_data[i,j,k]
     if val >0 :
        vN1 = PrelSegm_data[i-1,j,k]
        vN2 = PrelSegm_data[i+1,j,k]
        vN3 = PrelSegm_data[i,j-1,k]
        vN4 = PrelSegm_data[i,j+1,k]
        vN5 = PrelSegm_data[i,j,k-1]
        vN6 = PrelSegm_data[i,j,k+1]
        if PrelSegm_data[i,j,k] == 255 and ACSF_data[i,j,k] >= 26 :
          count1 = 0
          if vN1 <> 255 :
             count1 = count1 + 1
          if vN2 <> 255 :
             count1 = count1 + 1
          if vN3 <> 255 :
             count1 = count1 + 1
          if vN4 <> 255 :
             count1 = count1 + 1
          if vN5 <> 255 :
             count1 = count1 + 1
          if vN6 <> 255 :
             count1 = count1 + 1
          wmN = vN1==190 or vN2==190 or vN3==190 or vN4==190 or vN5==190 or vN6==190
          if count1 >= 4 and wmN :
             PrelSegm_data[i,j,k] = 190
             changesCount = changesCount + 1
 print "uwm changes={}".format(changesCount)
 if changesCount == 0 :
    break



Im = Image.fromarray(np.uint8(PrelSegm_data[:,:,nSlice]))
Im.save(results_dir+'SegmCGMEnhanced.png')
Im = Image.fromarray(np.uint8(PrelSegm_data[:,:,nSlice+difSlice]))
Im.save(results_dir+'SegmCGMEnhanced_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(PrelSegm_data[:,:,nSlice-difSlice]))
Im.save(results_dir+'SegmCGMEnhanced_m'+str(difSlice)+'.png')

Im = Image.fromarray(np.uint8(GT_data[:,:,nSliceGT]*30))
Im.save(results_dir+'SegmGT.png')
Im = Image.fromarray(np.uint8(GT_data[:,:,nSliceGT+difSliceGT]*30))
Im.save(results_dir+'SegmGT_p'+str(difSliceGT)+'.png')
Im = Image.fromarray(np.uint8(GT_data[:,:,nSliceGT-difSliceGT]*30))
Im.save(results_dir+'SegmGT_m'+str(difSliceGT)+'.png')
print "Until Cortical GM enhancing segmentation = {}.".format(time.time() - start_time)


#   Enhance segmentation of Subcortical Gray Matter zone based on
#   the EM algorithm (Makropoulos, 2015).

#     Generate SGM, UWM, CSF and CGM priors on Subcortical Gray Matter zone.
#     ----------------------------------------------------------------------
#     Find SGM zone bounding box.
maxISGM = 0
maxJSGM = 0
maxKSGM = 0
minISGM = dim1
minJSGM = dim2
minKSGM = dim3
feature_vSGM = []
for i in range(minI,maxI+1):
 for j in range(minJ,maxJ+1):
  for k in range(minK,maxK+1):
     if t2CurrentSubject_data[i,j,k] > 0 and ADGM_data[i,j,k] > 51 :
        feature_vSGM.append(t2CurrentSubject_data[i,j,k])
        if i < minISGM :
           minISGM = i
        if j < minJSGM :
           minJSGM = j
        if k < minKSGM :
           minKSGM = k
        if i > maxISGM :
           maxISGM = i
        if j > maxJSGM :
           maxJSGM = j
        if k > maxKSGM :
           maxKSGM = k

#     ----------------------------------------------------------------------
#     Cluster SGM zone to 3 classes with K-means to generate priors.
from sklearn import cluster

np.random.seed(0)
feature_vSGM = np.array(feature_vSGM,dtype=float)
k_meansSGM   = cluster.KMeans(n_clusters=3)
k_meansSGM.fit(feature_vSGM.reshape((feature_vSGM.shape[0],1)))
labelsSGM    = k_meansSGM.labels_
centersSGM   = sort(k_meansSGM.cluster_centers_,axis=0)

dim1SGM = maxISGM-minISGM+1
dim2SGM = maxJSGM-minJSGM+1
dim3SGM = maxKSGM-minKSGM+1
UWMPrior  = np.array(np.zeros((dim1SGM,dim2SGM,dim3SGM)),dtype=float)
CSFPrior  = np.array(np.zeros((dim1SGM,dim2SGM,dim3SGM)),dtype=float)
CGMPrior  = np.array(np.zeros((dim1SGM,dim2SGM,dim3SGM)),dtype=float)
SGMPrior  = np.array(np.zeros((dim1SGM,dim2SGM,dim3SGM)),dtype=float)

for i in range(minISGM,maxISGM+1):
 for j in range(minJSGM,maxJSGM+1):
  for k in range(minKSGM,maxKSGM+1):
     val = t2CurrentSubject_data[i,j,k]
     if val > 0 and ADGM_data[i,j,k] > 51 :
       m1 = (val - centersSGM[0,0])**2
       m2 = (val - centersSGM[1,0])**2
       m3 = (val - centersSGM[2,0])**2
       minM = min(min(m1,m2),m3)
       if minM == m3 : # CSF
          CSFPrior[i-minISGM,j-minJSGM,k-minKSGM] = val
       elif minM == m2 : # UWM
          UWMPrior[i-minISGM,j-minJSGM,k-minKSGM] = val
       else : # GM
          SGMPrior[i-minISGM,j-minJSGM,k-minKSGM] = val

muCSF  = np.mean(CSFPrior[ CSFPrior>0 ])
muUWM  = np.mean(UWMPrior[ UWMPrior>0 ])
muSGM  = np.mean(SGMPrior[ SGMPrior>0 ])
stdCSF = np.std(CSFPrior[ CSFPrior>0 ])
stdUWM = np.std(UWMPrior[ UWMPrior>0 ])
stdSGM = np.std(SGMPrior[ SGMPrior>0 ])

CSFPrior = GaussianVect(CSFPrior,muCSF,stdCSF)
UWMPrior = GaussianVect(UWMPrior,muUWM,stdUWM)
SGMPrior = GaussianVect(SGMPrior,muSGM,stdSGM)

#     Blur each K-means prior with Gaussian kernel with standard
#     deviation 1.
UWMPrior = NormalizeIntensity(ndimage.filters.gaussian_filter(UWMPrior,sigma=1.0),1.0)
CSFPrior = NormalizeIntensity(ndimage.filters.gaussian_filter(CSFPrior,sigma=1.0),1.0)
SGMPrior = NormalizeIntensity(ndimage.filters.gaussian_filter(SGMPrior,sigma=1.0),1.0)
muCSF  = 0
muUWM  = 0
muSGM  = 0
muCGM  = 0
stdCSF = 0
stdUWM = 0
stdSGM = 0
stdCGM = 0
nCSF   = 0
nUWM   = 0
nSGM   = 0
nCGM   = 0
#     Combine K-means with atlas information to improve the priors.
for i in range(minISGM,maxISGM+1):
 for j in range(minJSGM,maxJSGM+1):
  for k in range(minKSGM,maxKSGM+1):
     valT2 = t2CurrentSubject_data[i,j,k] / 255.0
     if valT2 > 0 and ADGM_data[i,j,k] > 51 :
       if A50_data[i,j,k] >= 1 and A50_data[i,j,k] <= 4 : #CGM
          CGMPrior[i-minISGM,j-minJSGM,k-minKSGM] = SGMPrior[i-minISGM,j-minJSGM,k-minKSGM]
          SGMPrior[i-minISGM,j-minJSGM,k-minKSGM] = 1.0-SGMPrior[i-minISGM,j-minJSGM,k-minKSGM]
          nCGM = nCGM + 1
          muCGM = muCGM + valT2 #CGMPrior[i-minISGM,j-minJSGM,k-minKSGM]
       else :
          #SGMPrior[i-minISGM,j-minJSGM,k-minKSGM] = SGMPrior[i-minISGM,j-minJSGM,k-minKSGM]*(ADGM_data[i,j,k]/255.0)
          SGMPrior[i-minISGM,j-minJSGM,k-minKSGM] = ADGM_data[i,j,k]/255.0
          nSGM = nSGM + 1
          muSGM = muSGM + valT2 #SGMPrior[i-minISGM,j-minJSGM,k-minKSGM]
          #UWMPrior[i-minISGM,j-minJSGM,k-minKSGM] = UWMPrior[i-minISGM,j-minJSGM,k-minKSGM]*(AWM_data[i,j,k]/255.0)
          UWMPrior[i-minISGM,j-minJSGM,k-minKSGM] = AWM_data[i,j,k]/255.0
          if AWM_data[i,j,k]/255.0 > 0 :
             nUWM = nUWM + 1
             muUWM = muUWM + valT2 #UWMPrior[i-minISGM,j-minJSGM,k-minKSGM]
          CSFPrior[i-minISGM,j-minJSGM,k-minKSGM] = CSFPrior[i-minISGM,j-minJSGM,k-minKSGM]*(ACSF_data[i,j,k]/255.0)
          if CSFPrior[i-minISGM,j-minJSGM,k-minKSGM] > 0 :
             nCSF = nCSF + 1
             muCSF = muCSF + valT2 #CSFPrior[i-minISGM,j-minJSGM,k-minKSGM]


muCGM = muCGM / nCGM
muSGM = muSGM / nSGM
muUWM = muUWM / nUWM
muCSF = muCSF / nCSF
for i in range(minISGM,maxISGM+1):
 for j in range(minJSGM,maxJSGM+1):
  for k in range(minKSGM,maxKSGM+1):
     valT2 = t2CurrentSubject_data[i,j,k] / 255.0
     if valT2 > 0 and ADGM_data[i,j,k] > 51 :
       if A50_data[i,j,k] >= 1 and A50_data[i,j,k] <= 4 : #CGM
          stdCGM = stdCGM + (valT2 - muCGM)**2 #(CGMPrior[i,j,k] - muCGM)**2
       else :
          stdSGM = stdSGM + (valT2 - muSGM)**2  #(SGMPrior[i,j,k] - muSGM)**2
          if AWM_data[i,j,k]/255.0 > 0 :
             stdUWM = stdUWM + (valT2 - muUWM)**2  #(UWMPrior[i,j,k] - muUWM)**2
          if CSFPrior[i-minISGM,j-minJSGM,k-minKSGM] > 0 :
             stdCSF = stdCSF + (valT2 - muCSF)**2  #(CSFPrior[i,j,k] - muCSF)**2

stdCGM = math.sqrt(stdCGM / (nCGM-1))
stdSGM = math.sqrt(stdSGM / (nSGM-1))
stdUWM = math.sqrt(stdUWM / (nUWM-1))
stdCSF = math.sqrt(stdCSF / (nCSF-1))


print "min(UWMPrior)={}, max(UWMPrior)={}".format(UWMPrior.min(),UWMPrior.max())
print "min(CSFPrior)={}, max(CSFPrior)={}".format(CSFPrior.min(),CSFPrior.max())
print "min(SGMPrior)={}, max(SGMPrior)={}".format(SGMPrior.min(),SGMPrior.max())


print "Until generation of priors for SGM zone = {}.".format(time.time() - start_time)

#     ----------------------------------------------------------------------
#     Apply (Makropoulos, 2015) EM algorithm (without correction step).
#     Code for each class: 0 - CSF, 1 - UWM, 2 - CGM, 3 - SGM)

#     Initialize parameters
K      = 4
tole   = 1.0E-4
maxIte = 5
betha  = 0.33
sx     = 1.0 / n_zooms[0]
sy     = sx
sz     = sx
pi     = np.array(np.zeros((dim1SGM,dim2SGM,dim3SGM,K)),dtype=float)
mu     = np.array(np.zeros(K),dtype=float)
muNew  = np.array(np.zeros(K),dtype=float)
std    = np.array(np.zeros(K),dtype=float)
stdNew = np.array(np.zeros(K),dtype=float)
muNew[0]  = muCSF
muNew[1]  = muUWM
muNew[2]  = muCGM
muNew[3]  = muSGM
stdNew[0] = stdCSF
stdNew[1] = stdUWM
stdNew[2] = stdCGM
stdNew[3] = stdSGM
print "mean(CSF)={},std(CSF)={}".format(muNew[0],stdNew[0])
print "mean(UWM)={},std(UWM)={}".format(muNew[1],stdNew[1])
print "mean(CGM)={},std(CGM)={}".format(muNew[2],stdNew[2])
print "mean(SGM)={},std(SGM)={}".format(muNew[3],stdNew[3])
pi[:,:,:,0] = CSFPrior
pi[:,:,:,1] = UWMPrior
pi[:,:,:,2] = CGMPrior
pi[:,:,:,3] = SGMPrior
for i in range(minISGM+1,maxISGM+0):
 for j in range(minJSGM+1,maxJSGM+0):
  for k in range(minKSGM+1,maxKSGM+0):
    i1 = i-minISGM
    j1 = j-minJSGM
    k1 = k-minKSGM
    if pi[i1,j1,k1,0]<0.1 and pi[i1,j1,k1,1]<0.1 and pi[i1,j1,k1,2]<0.1 and pi[i1,j1,k1,3]<0.1 :
       pi[i1,j1,k1,0] = 0.1
       pi[i1,j1,k1,1] = 0.1
       pi[i1,j1,k1,2] = 0.1
       pi[i1,j1,k1,3] = 0.1
del CSFPrior
del UWMPrior
del CGMPrior
del SGMPrior
p           = pi


#     Apply iterative EM algorithm
it = 0
normMu  = np.linalg.norm(muNew-mu)
normSTD = np.linalg.norm(stdNew-std)
while it < maxIte : #and normMu > tole and normSTD > tole :
   mu = muNew
   std = stdNew
   # Expectation step
   for i in range(minISGM+1,maxISGM+0):
    for j in range(minJSGM+1,maxJSGM+0):
     for k in range(minKSGM+1,maxKSGM+0):
       i1 = i-minISGM
       j1 = j-minJSGM
       k1 = k-minKSGM
       t2Val = t2CurrentSubject_data[i,j,k] / 255.0
       denom = 0
       if t2Val > 0 and ADGM_data[i,j,k] > 51 :
         for k2 in range(0,K):
            G     = Gaussian(t2Val,mu[k2],std[k2])
            denom = denom + pi[i1,j1,k1,k2]*math.exp(-betha*UMRF(k2,K,p,i1,j1,k1,sx,sy,sz))*G
         for k2 in range(0,K):
            G     = Gaussian(t2Val,mu[k2],std[k2])
            p[i1,j1,k1,k2] = pi[i1,j1,k1,k2]*math.exp(-betha*UMRF(k2,K,p,i1,j1,k1,sx,sy,sz))*G / denom
   # Maximization step
   for k2 in range(0,K):
     denom = 0
     nume  = 0
     for i in range(minISGM+1,maxISGM+0):
      for j in range(minJSGM+1,maxJSGM+0):
       for k in range(minKSGM+1,maxKSGM+0):
         i1 = i-minISGM
         j1 = j-minJSGM
         k1 = k-minKSGM
         t2Val = t2CurrentSubject_data[i,j,k] / 255.0
         if t2Val > 0 and ADGM_data[i,j,k] > 51 :
           denom = denom + p[i1,j1,k1,k2]
           nume  = nume  + p[i1,j1,k1,k2]*t2Val
     muNew[k2] = nume / denom
     nume  = 0
     for i in range(minISGM+1,maxISGM+0):
      for j in range(minJSGM+1,maxJSGM+0):
       for k in range(minKSGM+1,maxKSGM+0):
         i1 = i-minISGM
         j1 = j-minJSGM
         k1 = k-minKSGM
         t2Val = t2CurrentSubject_data[i,j,k] / 255.0
         if t2Val > 0 and ADGM_data[i,j,k] > 51 :
           nume  = nume  + p[i1,j1,k1,k2] * (t2Val-muNew[k2])**2
     stdNew[k2] = math.sqrt(nume / denom)
   normMu  = np.linalg.norm(muNew-mu)
   normSTD = np.linalg.norm(stdNew-std)
   it = it + 1
   print "EM. Ite={}, normMuDif = {}, normSTDDif = {}.".format(it,normMu,normSTD)
   print "mean(CSF)={},std(CSF)={}".format(muNew[0],stdNew[0])
   print "mean(UWM)={},std(UWM)={}".format(muNew[1],stdNew[1])
   print "mean(CGM)={},std(CGM)={}".format(muNew[2],stdNew[2])
   print "mean(SGM)={},std(SGM)={}".format(muNew[3],stdNew[3])

#     Final SGM zone labeling of voxels according to EM
#     result after convergence.
for i in range(minISGM+1,maxISGM+0):
 for j in range(minJSGM+1,maxJSGM+0):
  for k in range(minKSGM+1,maxKSGM+0):
    i1 = i-minISGM
    j1 = j-minJSGM
    k1 = k-minKSGM
    if t2CurrentSubject_data[i,j,k] > 0 and ADGM_data[i,j,k] > 51 :
      maxAPost = max(max(max(p[i1,j1,k1,0],p[i1,j1,k1,1]),p[i1,j1,k1,2]),p[i1,j1,k1,3])
      if maxAPost == p[i1,j1,k1,0] : #CSF
         PrelSegm_data[i,j,k] = 255
      elif maxAPost == p[i1,j1,k1,1] : #UWM
         PrelSegm_data[i,j,k] = 190
      elif maxAPost == p[i1,j1,k1,2] : #CGM
         PrelSegm_data[i,j,k] = 85
      else : #SGM
         PrelSegm_data[i,j,k] = 50


del p
del pi

Im = Image.fromarray(np.uint8(PrelSegm_data[:,:,nSlice]))
Im.save(results_dir+'SegmCGMEnhanced.png')
Im = Image.fromarray(np.uint8(PrelSegm_data[:,:,nSlice+difSlice]))
Im.save(results_dir+'SegmCGMEnhanced_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(PrelSegm_data[:,:,nSlice-difSlice]))
Im.save(results_dir+'SegmCGMEnhanced_m'+str(difSlice)+'.png')

print "Until SGM zone enhancing = {}.".format(time.time() - start_time)




#   Obtain DICE coefficient for UWM, CGM, SGM and CSF.
#   Ground Truth clases (3D array GT_data):
#      1 - Cortical GM
#      2 - Basal ganglia and Thalami (Subcortical GM)
#      3 - Unmyelinated WM
#      4 - Myelinated WM                             --> *Removed
#      5 - Brainstem (without myelinated penduncies) --> *Removed
#      6 - Cerebellum                                --> *Removed
#      7 - Ventricles (lateral, third and fourth, this is ventricular CSF)
#      8 - External CSF (CSF that is not ventricular)

numeCGM  = 0
numeSGM  = 0
numeWM   = 0
numeCSF  = 0
denSMCGM = 0
denSMSGM = 0
denSMWM  = 0
denSMCSF = 0
denGTCGM = 0
denGTSGM = 0
denGTWM  = 0
denGTCSF = 0
#wVoxel   = int(round(float(dim3)/float(GT_data.shape[2])))

for i in range(0,dim1):
 for j in range(0,dim2):
  for k in range(0,GT_data.shape[2]):
   if GT_data[i,j,k] <> 6 and GT_data[i,j,k] <> 5 and GT_data[i,j,k]<>4 :
     k1 = int(round(float(dim3*k)/float(GT_data.shape[2])))
     # decide voxel value on third dimension based on voting scheme
     #GMVotes  = 0
     #WMVotes  = 0
     #CSFVotes = 0
     #segValue = 0
     #for k2 in xrange(0,wVoxel):
     #   ind = k1 + k2
     #   if ind < dim3 :
     #      if SegMap[i,j,ind] == 85 :
     #         GMVotes = GMVotes + 1
     #      elif SegMap[i,j,ind] == 255 :
     #         CSFVotes = CSFVotes + 1
     #      else:
     #         if SegMap[i,j,ind] == 190 :
     #            WMVotes = WMVotes + 1
     #maxVoted = max(max(GMVotes,CSFVotes),WMVotes)
     #if maxVoted == 0 :
     #   segValue = 0
     #elif maxVoted == GMVotes :
     #   segValue = 85
     #elif maxVoted == CSFVotes :
     #   segValue = 255
     #else :
     #   segValue = 190
     segValue = PrelSegm_data[i,j,k1]
     
     if segValue == 85 :
        denSMCGM  = denSMCGM + 1
     elif segValue == 50 :
        denSMSGM = denSMSGM + 1
     elif segValue == 255 :
        denSMCSF = denSMCSF + 1
     else :
        if segValue == 190 :
           denSMWM  = denSMWM + 1
     if GT_data[i,j,k] == 1 :
        denGTCGM  = denGTCGM + 1
        if segValue == 85 :
           numeCGM   = numeCGM + 1
     elif GT_data[i,j,k] == 2 :
        denGTSGM  = denGTSGM + 1
        if segValue == 50 :
           numeSGM   = numeSGM + 1
     elif GT_data[i,j,k] == 3 :
        denGTWM  = denGTWM + 1
        if segValue == 190 :
           numeWM   = numeWM + 1
     else :
        if GT_data[i,j,k] == 7 or GT_data[i,j,k] == 8 :
           denGTCSF = denGTCSF + 1
           if segValue == 255 :
              numeCSF = numeCSF + 1


DICE_CGM  = 2.0*numeCGM  / (denSMCGM  + denGTCGM)
DICE_SGM  = 2.0*numeSGM  / (denSMSGM  + denGTSGM)
DICE_WM  = 2.0*numeWM  / (denSMWM  + denGTWM)
DICE_CSF = 2.0*numeCSF / (denSMCSF + denGTCSF)

print "DICE CGM = {}".format(DICE_CGM)
print "DICE SGM = {}".format(DICE_SGM)
print "DICE UWM = {}".format(DICE_WM)
print "DICE CSF = {}".format(DICE_CSF)


#   save results

#SegMapVolume = nib.Nifti1Image(SegMap,affineT2CS)
#nib.save(SegMapVolume,t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/SegMapVolumeCGM.nii.gz")

