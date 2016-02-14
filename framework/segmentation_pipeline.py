import numpy as np
import scipy as sp
import nibabel as nib
import os
import argparse
import time
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *
from skimage.morphology import ball,opening,closing,dilation,erosion,watershed
from scipy.ndimage.measurements import watershed_ift
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_bilateral
from scipy.stats.stats import pearsonr
from dipy.align.reslice import reslice
import scipy.ndimage as ndimage

# Normalize intensity values to the range [0,scaleVal]
# Input:  data numpy array with the intensity values to normalize and
#         scaleVal value for the normalization.
# Output: data numpy array with the normalized intensity values.
def NormalizeIntensity(data,scaleVal):
   maxVal = data.max()
   minVal = data.min()
   data = (scaleVal/(maxVal-minVal))*(data-minVal)
   return data


#base_dir = '/home/omar/data/DATA_NeoBrainS12/'
base_dir    = '/home/dalmau/opt/imagenes/data_NeoBrainS12/'
neo_subject = '30wCoronal/example2/'
results_dir = '/home/dalmau/opt/segmentation/framework/segmentation_pipeline_output/example2_30wCoronal/'
#results_dir = '/home/omar/opt/segmentation/framework/segmentation_pipeline_output/example2_30wCoronal/'
atlas_label = '28w'

# Read subject files
t2CurrentSubjectName  = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'T2_1-1.nii.gz'
t1CurrentSubjectName  = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'T1_1-2.nii.gz'
GTName                = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'manualSegm.nii.gz'
t2CurrentSubject_data = nib.load(t2CurrentSubjectName).get_data()
t1CurrentSubject_data = nib.load(t1CurrentSubjectName).get_data()
GT_data               = nib.load(GTName).get_data()
affineT2CS            = nib.load(t2CurrentSubjectName).get_affine()
zoomsT2CS             = nib.load(t2CurrentSubjectName).get_header().get_zooms()[:3]


# Read priors files
AT1Name    = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'A'+atlas_label+'_T1.nii.gz'
AT2Name    = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'A'+atlas_label+'_T2.nii.gz'
AMaskName  = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'A'+atlas_label+'_Mask.nii.gz'
ABSName    = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'A'+atlas_label+'_BS.nii.gz'
ACeName    = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'A'+atlas_label+'_Ce.nii.gz'
ACoName    = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'A'+atlas_label+'_Co.nii.gz'
ACSFName   = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'A'+atlas_label+'_CSF.nii.gz'
ADGMName   = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'A'+atlas_label+'_DGM.nii.gz'
AWMName    = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'A'+atlas_label+'_WM.nii.gz'
AT1_data   = nib.load(AT1Name).get_data()
AT2_data   = nib.load(AT2Name).get_data()
AMask_data = nib.load(AMaskName).get_data()
ABS_data   = nib.load(ABSName).get_data()
ACe_data   = nib.load(ACeName).get_data()
ACo_data   = nib.load(ACoName).get_data()
ACSF_data  = nib.load(ACSFName).get_data()
ADGM_data  = nib.load(ADGMName).get_data()
AWM_data   = nib.load(AWMName).get_data()


start_time = time.time()

# Step 1.4 - Resampling for isotropic voxels

n_zooms = (zoomsT2CS[0],zoomsT2CS[0],zoomsT2CS[0])
v = n_zooms[0]
t2CurrentSubject_data,affineT2CS = reslice(t2CurrentSubject_data,affineT2CS,zoomsT2CS,n_zooms)
t1CurrentSubject_data,_          = reslice(t1CurrentSubject_data,affineT2CS,zoomsT2CS,n_zooms)
AT1_data,_                       = reslice(AT1_data,affineT2CS,zoomsT2CS,n_zooms)
AT2_data,_                       = reslice(AT2_data,affineT2CS,zoomsT2CS,n_zooms)
AMask_data,_                     = reslice(AMask_data,affineT2CS,zoomsT2CS,n_zooms)
ABS_data,_                       = reslice(ABS_data,affineT2CS,zoomsT2CS,n_zooms)
ACe_data,_                       = reslice(ACe_data,affineT2CS,zoomsT2CS,n_zooms)
ACo_data,_                       = reslice(ACo_data,affineT2CS,zoomsT2CS,n_zooms)
ACSF_data,_                      = reslice(ACSF_data,affineT2CS,zoomsT2CS,n_zooms)
ADGM_data,_                      = reslice(ADGM_data,affineT2CS,zoomsT2CS,n_zooms)
AWM_data,_                       = reslice(AWM_data,affineT2CS,zoomsT2CS,n_zooms)

# Step 1.5 - Anisotropic diffusion filter

scaleValue = 1.0
t2CurrentSubject_data = denoise_bilateral(NormalizeIntensity(t2CurrentSubject_data,scaleValue))
t1CurrentSubject_data = denoise_bilateral(NormalizeIntensity(t1CurrentSubject_data,scaleValue))


# Normalize the rest of the volume intensity values to [0,255]
scaleValue            = 255.0
t2CurrentSubject_data = NormalizeIntensity(t2CurrentSubject_data,scaleValue)
t1CurrentSubject_data = NormalizeIntensity(t1CurrentSubject_data,scaleValue)
AT1_data              = NormalizeIntensity(AT1_data,scaleValue)
AT2_data              = NormalizeIntensity(AT2_data,scaleValue)
AMask_data            = NormalizeIntensity(AMask_data,scaleValue)
ABS_data              = NormalizeIntensity(ABS_data,scaleValue)
ACe_data              = NormalizeIntensity(ACe_data,scaleValue)
ACo_data              = NormalizeIntensity(ACo_data,scaleValue)
ACSF_data             = NormalizeIntensity(ACSF_data,scaleValue)
ADGM_data             = NormalizeIntensity(ADGM_data,scaleValue)
AWM_data              = NormalizeIntensity(AWM_data,scaleValue)

dim1 = t2CurrentSubject_data.shape[0]
dim2 = t2CurrentSubject_data.shape[1]
dim3 = t2CurrentSubject_data.shape[2]


# 2 - Intracranial Cavity Extraction

#   apply atlas head mask
t2CurrentSubject_data[AMask_data == 0] = 0
#for i in xrange(0,dim1):
#  for j in xrange(0,dim2):
#   for k in xrange(0,dim3):
#      if AMask_data[i,j,k] == 0 :
#        t2CurrentSubject_data[i,j,k] = 0

nSlice = int(dim3 / 2)
difSlice = 12

Im = Image.fromarray(np.uint8(t2CurrentSubject_data[:,:,nSlice]))
Im.save(results_dir+'T2_IC.png')
Im = Image.fromarray(np.uint8(t2CurrentSubject_data[:,:,nSlice+difSlice]))
Im.save(results_dir+'T2_IC_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(t2CurrentSubject_data[:,:,nSlice-difSlice]))
Im.save(results_dir+'T2_IC_m'+str(difSlice)+'.png')

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
dilGradOT2   = dilation(gradientOT2,ball(0))
dilGradOT2   = dilation(gradientOT2,ball(1)) + dilGradOT2
dilGradOT2   = dilation(gradientOT2,ball(2)) + dilGradOT2
dilGradOT2   = dilation(gradientOT2,ball(3)) + dilGradOT2
dilGradOT2   = dilation(gradientOT2,ball(4)) + dilGradOT2
segFuncGOT2  = NormalizeIntensity(dilGradOT2,255.0)
Im = Image.fromarray(np.uint8(segFuncGOT2[:,:,nSlice]))
Im.save(results_dir+'seg_func_ICE.png')
del dilGradOT2
del gradientOT2

#   Obtain T2 mask by threshold
#t = threshold_otsu(t2CurrentSubject_data)
#maskT2 = t2CurrentSubject_data >= (t*1.75) # 2.1)
#maskT2 = np.array(maskT2,dtype=float)

#   Obtain gravity center of mask of T2
current_mask = (t2CurrentSubject_data>0).astype(np.int32)

C = ndimage.measurements.center_of_mass(current_mask)
maskT2Count = np.sum(current_mask)

#C = np.zeros(3)
#maskT2Count = 0
#for x in xrange(0,dim1):
#  for y in xrange(0,dim2):
#    for z in xrange(0,dim3):
#       #if maskT2[x,y,z] > 0 :
#       if t2CurrentSubject_data[x,y,z] > 0 :
#          maskT2Count = maskT2Count + 1
#          C[0] = C[0] + x
#          C[1] = C[1] + y
#          C[2] = C[2] + z

#C = C / float(maskT2Count)
print "Centroid = {}".format(C)

#   set two class of markers (for marker based watershed segmentation)
markersICE = np.array(np.zeros((dim1,dim2,dim3)),dtype=int)
markersICE[int(C[0]),int(C[1]),int(C[2])] = 2
for i in xrange(1,4):
  markersICE[int(C[0])+i,int(C[1]),int(C[2])] = 2
  markersICE[int(C[0])-i,int(C[1]),int(C[2])] = 2
  markersICE[int(C[0]),int(C[1])+i,int(C[2])] = 2
  markersICE[int(C[0]),int(C[1])-i,int(C[2])] = 2
  markersICE[int(C[0]),int(C[1]),int(C[2])+i] = 2
  markersICE[int(C[0]),int(C[1]),int(C[2])-i] = 2

for y in xrange(0,dim2):
  for z in xrange(0,dim3):
     markersICE[0,y,z] = 1
     markersICE[dim1-1,y,z] = 1

for x in xrange(0,dim1):
  for z in xrange(0,dim3):
     markersICE[x,0,z] = 1
     markersICE[x,dim2-1,z] = 1

for y in xrange(0,dim2):
  for x in xrange(0,dim1):
     markersICE[x,y,0] = 1
     markersICE[x,y,dim3-1] = 1

#   Apply watershed segmentation with markers
segFuncGOT2 = np.array(segFuncGOT2,dtype=int)
ICEMask = watershed(segFuncGOT2,markersICE)
del segFuncGOT2
del markersICE
ICEMask = dilation(ICEMask,ball(1))
#   Apply Inctracranial Cavity Extraction with segmented watershed mask
t2CurrentSubject_data[ICEMask == 1] = 0
t1CurrentSubject_data[ICEMask == 1] = 0
#for x in xrange(0,dim1):
#  for y in xrange(0,dim2):
#    for z in xrange(0,dim3):
#       if ICEMask[x,y,z] == 1 :
#          t2CurrentSubject_data[x,y,z] = 0
#          t1CurrentSubject_data[x,y,z] = 0

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


print "Until ICE: {} seconds.".format(time.time() - start_time)


del ICEMask


# Hemisphere Separation

#   Obtain correlation coefficients
#corrCoef = np.zeros((dim1,dim2))
#z = nSlice
#w = 7 # must be odd
#w_2 = math.floor(w/2)

#for i in xrange(w,dim1-w):
# for j in xrange(w,dim2-w):
#  if t2CurrentSubject_data[i,j,z] > 0 :
#     # Get cuboid for current voxel
#     u_1 = np.zeros(w**3 -1)
#     u_2 = np.zeros(w**3 -1)
#     ind = 0
#     for k1 in xrange(-int(w_2),int(w_2)):
#       for k2 in xrange(-int(w_2),int(w_2)):
#         for k3 in xrange(-int(w_2),int(w_2)):
#           i1 = i+k1
#           j1 = j+k2
#           z1 = z+k3
#           if k1<>0 and k2<>0 and k3<>0 :
#              u_1[ind] = t2CurrentSubject_data[i1,j1,z1]
#              u_2[ind] = t2CurrentSubject_data[dim1-i1-1,j1,z1]
#              ind = ind + 1
#     pearson = pearsonr(u_1, u_2)
#     corrCoef[i,j] = pearson[0]
#     if np.isnan(corrCoef[i,j]) or corrCoef[i,j] < 0 :
#        corrCoef[i,j] = 0

#Im = Image.fromarray(np.uint8(corrCoef[:,:]*255.0))
#Im.save('corrCoef.png')

#   set markers
#markersHem = np.array(np.zeros((dim1,dim2)),dtype=int)

#for y in xrange(0,dim2):
#  #for z in xrange(0,dim3):
#  markersHem[int(C[0])-40,y] = 1
#  markersHem[int(C[0])+40,y] = 2

#   Apply watershed segmentation with markers
#segFuncHemT2_int = np.array(corrCoef*255.0,dtype=int)
#HemMask = watershed(segFuncHemT2_int,markersHem)

#Im = Image.fromarray(np.uint8(HemMask*127))
#Im.save('HemSepMask.png')


# 3 - Detection of GM, UWM and CSF

# Step 3.1 - Subcortical Grey Matter enhancing

#   large scale closing of T2 image
#lsClosingT2 = closing(t2CurrentSubject_data,ball(10))
#Im = Image.fromarray(np.uint8(lsClosingT2[:,:,nSlice]))
#Im.save(results_dir+'lsClosingT2.png')
#Im = Image.fromarray(np.uint8(lsClosingT2[:,:,nSlice+difSlice]))
#Im.save(results_dir+'lsClosingT2_p'+str(difSlice)+'.png')
#Im = Image.fromarray(np.uint8(lsClosingT2[:,:,nSlice-difSlice]))
#Im.save(results_dir+'lsClosingT2_m'+str(difSlice)+'.png')
#del lsClosingT2

#   sum of increasing scale closings of T2
# Note (JOOG): the running time increases with the size of the structuring
# element (which is "flat" in this case), by any chance do you know what is the
# complexity of the algorithm implemented in scikit image?. It appears to me
# that there should be a faster way of doing this, don't you think Oscar,
# Ulises? According to this:
# http://www.ee.lamar.edu/gleb/dip/10-3%20-%20Morphological%20Image%20Processing.pdf
# erosion just picks the minimum value of the input image within the
# structuring element. And dilation takes the maximum.
sumClosingT2 = closing(t2CurrentSubject_data,ball(0))
for r in xrange(1,11):
   print("Iter: ",r)
   sumClosingT2 = sumClosingT2 + closing(t2CurrentSubject_data,ball(r))

print "closing 1-10 done."

for r in xrange(11,13):
   sumClosingT2 = sumClosingT2 + closing(t2CurrentSubject_data,ball(r))

print "closing 11-12 done."

for r in xrange(13,17):
   sumClosingT2 = sumClosingT2 + closing(t2CurrentSubject_data,ball(r))

print "closing 13-16 done."

for r in xrange(17,19):
   sumClosingT2 = sumClosingT2 + closing(t2CurrentSubject_data,ball(r))


Im = Image.fromarray(np.uint8(NormalizeIntensity(sumClosingT2[:,:,nSlice],255.0)))
Im.save(results_dir+'sumClosingT2.png')
Im = Image.fromarray(np.uint8(NormalizeIntensity(sumClosingT2[:,:,nSlice+difSlice],255.0)))
Im.save(results_dir+'sumClosingT2_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(NormalizeIntensity(sumClosingT2[:,:,nSlice-difSlice],255.0)))
Im.save(results_dir+'sumClosingT2_m'+str(difSlice)+'.png')


print "Until SGM Segmentation function: {} seconds.".format(time.time() - start_time)

#   set markers
#markersSGM = np.array(np.zeros((dim1,dim2,dim3)),dtype=int)
#SGMMask = np.array(np.zeros((dim1,dim2,dim3)),dtype=int)
#for i in xrange(50,dim1-50):
# for j in xrange(50,dim2-50):
#  for k in xrange(50,dim3-50):
#    val = ADGM_data[i,j,k]
#    if val >217 :
#       #SGMMask[i,j,k] = 1
#       markersSGM[i,j,k] = 3



#markersSGM = erosion(markersSGM,ball(8))

#for i in xrange(10,dim1-10):
# for j in xrange(10,dim2-10):
#  for k in xrange(10,dim3-10):
#    if t2CurrentSubject_data[i,j,k] > 0 :
#     for l in xrange(1,7):
#       valN1 = t2CurrentSubject_data[i-l,j,k]
#       valN2 = t2CurrentSubject_data[i+l,j,k]
#       valN3 = t2CurrentSubject_data[i,j-l,k]
#       valN4 = t2CurrentSubject_data[i,j+l,k]
#       valN5 = t2CurrentSubject_data[i,j,k-l]
#       valN6 = t2CurrentSubject_data[i,j,k+l]
#       if valN1==0 or valN2==0 or valN3==0 or valN4==0 or valN5==0 or valN6==0 :
#          markersSGM[i,j,k] = 1


#Im = Image.fromarray(np.uint8(markersSGM[:,:,nSlice]*85))
#Im.save(results_dir+'markersSGM.png')

#   Apply watershed segmentation with markers
#sumClosingT2_int = np.array(NormalizeIntensity(sumClosingT2,255.0),dtype=np.uint16)
#sumClosingT2_int = np.array(NormalizeIntensity(lsClosingT2,255.0),dtype=np.uint16)
#SGMMask = watershed(sumClosingT2_int,markersSGM)
#SGMMask = watershed_ift(sumClosingT2_int,markersSGM)
#Im = Image.fromarray(np.uint8(SGMMask[:,:,nSlice]*85))
#Im.save(results_dir+'SGMMask.png')
#Im = Image.fromarray(np.uint8(SGMMask[:,:,nSlice+difSlice]*85))
#Im.save(results_dir+'SGMMask_p'+str(difSlice)+'.png')
#Im = Image.fromarray(np.uint8(SGMMask[:,:,nSlice-difSlice]*85))
#Im.save(results_dir+'SGMMask_m'+str(difSlice)+'.png')


# Step 3.2 - Preliminar segmentation (UWM, GM, CSF)

from sklearn import cluster

#   K-means clustering
np.random.seed(0)
feature_vector = []
for i in xrange(20,dim1-20):
 for j in xrange(20,dim2-20):
  for k in xrange(20,dim3-20):
    val = t2CurrentSubject_data[i,j,k]
    if val > 0 :
       feature_vector.append(val)

feature_vector = np.array(feature_vector,dtype=float)
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(feature_vector.reshape((feature_vector.shape[0],1)))
labels = k_means.labels_
l1 = sum(labels==0)
l2 = sum(labels==1)
l3 = sum(labels==2)
wmLab = max(l1,max(l2,l3))
if wmLab == l1 :
  wmLab = 0
elif wmLab == l2 :
  wmLab = 1
else :
  wmLab = 2

csfLab = min(l1,min(l2,l3))
if csfLab == l1 :
  csfLab = 0
elif csfLab == l2 :
  csfLab = 1
else :
  csfLab = 2

gmLab = 0
if wmLab == 0 :
  if csfLab == 1 :
    gmLab = 2
  else :
    gmLab = 1
elif wmLab == 1 :
  if csfLab == 0 :
    gmLab = 2
  else :
    gmLab = 0
else :
  if csfLab == 0 :
    gmLab = 1
  else :
    gmLab = 0


#values = k_means.cluster_centers_.squeeze()


#GMMap =  np.zeros((dim1,dim2,dim3),dtype=int)
#ind = 0
#for i in xrange(20,dim1-20):
# for j in xrange(20,dim2-20):
#  for k in xrange(20,dim3-20):
#    val = t2CurrentSubject_data[i,j,k]
#    if val >0 :
#       if labels[ind] == gmLab :
#          GMMap[i,j,k] = 255
#       ind = ind + 1

#Im = Image.fromarray(np.uint8(GMMap[:,:,nSlice]))
#Im.save(results_dir+'GMMap.png')
#GMMap1 = opening(GMMap,ball(3))
#Im = Image.fromarray(np.uint8(GMMap1[:,:,nSlice]))
#Im.save(results_dir+'GMMapOpening.png')

#   Get largest connected component of UWM resulted from kmeans and
#   excluding unlikely voxels according to Atlas
WMMap =  np.zeros((dim1,dim2,dim3),dtype=int)
ind = 0
for i in xrange(20,dim1-20):
 for j in xrange(20,dim2-20):
  for k in xrange(20,dim3-20):
    val = t2CurrentSubject_data[i,j,k]
    if val >0 :
       if labels[ind] == wmLab and AWM_data[i,j,k] >= 13 : #ADGM_data[i,j,k] < 230 :
          WMMap[i,j,k] = 255
       ind = ind + 1

Im = Image.fromarray(np.uint8(WMMap[:,:,nSlice]))
Im.save(results_dir+'WMMap.png')
WMMap = opening(WMMap,ball(4))
Im = Image.fromarray(np.uint8(WMMap[:,:,nSlice]))
Im.save(results_dir+'WMMapOpening.png')



#    uses default cross-shaped structuring element (for connectivity)
WLCC,numcomponents = sp.ndimage.measurements.label(WMMap)

if numcomponents > 1 :
  componentLabels = []
  for i_nc in xrange(0,numcomponents):
    labelId = i_nc + 1
    componentLabels.append( (sum(WLCC==labelId),labelId) )
  componentLabels.sort(key=lambda tup: tup[0],reverse=True)
  # keep only Largest Connected Component
  for i in xrange(0,dim1):
    for j in xrange(0,dim2):
      for k in xrange(0,dim3):
        if WLCC[i,j,k] == componentLabels[0][1] or WLCC[i,j,k] == componentLabels[1][1] :
           WLCC[i,j,k] = 255
        else :
           WLCC[i,j,k] = 0
else :
  WLCC = WLCC*255

Im = Image.fromarray(np.uint8(WLCC[:,:,nSlice]))
Im.save(results_dir+'WM_LCC.png')



#   Preliminar labeling of UWM based on Largest Connected Component,
#   GM based on atlas information and K-means clustering on regions
#   neighbouring UWM and CSF based on regions bordering background
#   and complement of the UWM Union GM.

SegMap =  np.zeros((dim1,dim2,dim3),dtype=int)
GMMap =  np.zeros((dim1,dim2,dim3),dtype=int)
feature_vector1 = []
ind = 0
for i in xrange(20,dim1-20):
 for j in xrange(20,dim2-20):
  for k in xrange(20,dim3-20):
    val = t2CurrentSubject_data[i,j,k]
    if val >0 :
       if WLCC[i,j,k] == 255 :
          SegMap[i,j,k] = 190
       elif labels[ind] == csfLab :
          SegMap[i,j,k] = 255
       else :
          SegMap[i,j,k] = 255
          if ADGM_data[i,j,k] >= 51 :
             feature_vector1.append(sumClosingT2[i,j,k])
             #feature_vector1.append(val*(1.1**(1.0- float(ADGM_data[i,j,k])/255.0))) #sumClosingT2[i,j,k])
             #SegMap[i,j,k] = 85
          elif ACSF_data[i,j,k] < 26 : #and ACo_data[i,j,k] >= 51 :
             SegMap[i,j,k] = 85
             GMMap[i,j,k] = 255
          else :
            if labels[ind] == gmLab :
               for l in xrange(1,7): #8):
                  valN1 = WLCC[i-l,j,k]
                  valN2 = WLCC[i+l,j,k]
                  valN3 = WLCC[i,j-l,k]
                  valN4 = WLCC[i,j+l,k]
                  valN5 = WLCC[i,j,k-l]
                  valN6 = WLCC[i,j,k+l]
                  if valN1==255 or valN2==255 or valN3==255 or valN4==255 or valN5==255 or valN6==255 :
                     SegMap[i,j,k] = 85
                     GMMap[i,j,k] = 255
          for l in xrange(1,5):
             valN1 = t2CurrentSubject_data[i-l,j,k]
             valN2 = t2CurrentSubject_data[i+l,j,k]
             valN3 = t2CurrentSubject_data[i,j-l,k]
             valN4 = t2CurrentSubject_data[i,j+l,k]
             valN5 = t2CurrentSubject_data[i,j,k-l]
             valN6 = t2CurrentSubject_data[i,j,k+l]
             if valN1==0 or valN2==0 or valN3==0 or valN4==0 or valN5==0 or valN6==0 :
                SegMap[i,j,k] = 255
                GMMap[i,j,k] = 0
       ind = ind + 1


Im = Image.fromarray(np.uint8(GMMap[:,:,nSlice]))
Im.save(results_dir+'GMMap.png')
Im = Image.fromarray(np.uint8(GMMap[:,:,nSlice+difSlice]))
Im.save(results_dir+'GMMap_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(GMMap[:,:,nSlice-difSlice]))
Im.save(results_dir+'GMMap_m'+str(difSlice)+'.png')


#   Filter undesirable CSF labeled inside UWM near border
#   regions with Cortical GM.

for rep in xrange(0,30):
 changesCount = 0
 for i in xrange(20,dim1-20):
  for j in xrange(20,dim2-20):
   for k in xrange(20,dim3-20):
     val = t2CurrentSubject_data[i,j,k]
     if val >0 :
        vN1 = SegMap[i-1,j,k]
        vN2 = SegMap[i+1,j,k]
        vN3 = SegMap[i,j-1,k]
        vN4 = SegMap[i,j+1,k]
        vN5 = SegMap[i,j,k-1]
        vN6 = SegMap[i,j,k+1]
        #if rep == 0 and SegMap[i,j,k] == 190 and ACo_data[i,j,k] >= 51 :
        #  gmN = vN1==85 or vN2==85 or vN3==85 or vN4==85 or vN5==85 or vN6==85
        #  if gmN :
        #     SegMap[i,j,k] = 190
        #     GMMap[i,j,k] = 0
        if SegMap[i,j,k] == 255 and ACSF_data[i,j,k] >= 26 :
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
             SegMap[i,j,k] = 190
             changesCount = changesCount + 1
 print "wm changes={}".format(changesCount)
 if changesCount == 0 :
    break


Im = Image.fromarray(np.uint8(SegMap[:,:,nSlice]))
Im.save(results_dir+'SegMap.png')
Im = Image.fromarray(np.uint8(SegMap[:,:,nSlice+difSlice]))
Im.save(results_dir+'SegMap_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(SegMap[:,:,nSlice-difSlice]))
Im.save(results_dir+'SegMap_m'+str(difSlice)+'.png')


#   Apply K-means on Subcortical GM atlas region but using the
#   information from enhanced Subcortical GM (sum of increasing
#   scale closings on T2 image) and keep only the two darker class
#   regions as Subcortical GM.

np.random.seed(0)
feature_vector1 = np.array(feature_vector1,dtype=float)
k_means1 = cluster.KMeans(n_clusters=3)
k_means1.fit(feature_vector1.reshape((feature_vector1.shape[0],1)))
labels1 = k_means1.labels_
centers1 = sort(k_means1.cluster_centers_,axis=0)
maxLabel1 = 0

ind = 0
ind1 = 0
for i in xrange(20,dim1-20):
 for j in xrange(20,dim2-20):
  for k in xrange(20,dim3-20):
    val = t2CurrentSubject_data[i,j,k]
    if val >0 :
       if WLCC[i,j,k] == 255 or labels[ind] == csfLab :
          val1 = 0
       else :
          if ADGM_data[i,j,k] >= 51 :
             val1 = sumClosingT2[i,j,k] #val*(1.1**(1.0- float(ADGM_data[i,j,k])/255.0)
             m1 = (val1 - centers1[0,0])**2
             m2 = (val1 - centers1[1,0])**2
             m3 = (val1 - centers1[2,0])**2
             #m4 = (val1 - centers1[3,0])**2
             #m5 = (val1 - centers1[4,0])**2
             #minM = min(min(min(min(m1,m2),m3),m4),m5)
             #if minM <> m5 and minM<> m4 :
             if min(min(m1,m2),m3) <> m3 :
                SegMap[i,j,k] = 85
             ind1 = ind1 + 1
       ind = ind + 1


Im = Image.fromarray(np.uint8(SegMap[:,:,nSlice]))
Im.save(results_dir+'SegMap1.png')
Im = Image.fromarray(np.uint8(SegMap[:,:,nSlice+difSlice]))
Im.save(results_dir+'SegMap1_p'+str(difSlice)+'.png')
Im = Image.fromarray(np.uint8(SegMap[:,:,nSlice-difSlice]))
Im.save(results_dir+'SegMap1_m'+str(difSlice)+'.png')


print "Until preliminar segmentation = {}.".format(time.time() - start_time)



#   Obtain DICE coefficient for UWM, GM and CSF.
#   Ground Truth clases (3D array GT_data):
#      1 - Cortical GM
#      2 - Basal ganglia and Thalami (Subcortical GM)
#      3 - Unmyelinated WM
#      4 - Myelinated WM                             --> *Removed
#      5 - Brainstem (without myelinated penduncies) --> *Removed
#      6 - Cerebellum                                --> *Removed
#      7 - Ventricles (lateral, third and fourth, this is ventricular CSF)
#      8 - External CSF (CSF that is not ventricular)

numeGM   = 0
numeWM   = 0
numeCSF  = 0
denSMGM  = 0
denSMWM  = 0
denSMCSF = 0
denGTGM  = 0
denGTWM  = 0
denGTCSF = 0
#wVoxel   = int(round(float(dim3)/float(GT_data.shape[2])))

for i in xrange(20,dim1-20):
 for j in xrange(20,dim2-20):
  for k in xrange(0,GT_data.shape[2]):
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
     segValue = SegMap[i,j,k1]

     if segValue == 85 :
        denSMGM  = denSMGM + 1
     elif segValue == 255 :
        denSMCSF = denSMCSF + 1
     else :
        if segValue == 190 :
           denSMWM  = denSMWM + 1
     if GT_data[i,j,k] == 1 or GT_data[i,j,k] == 2 :
        denGTGM  = denGTGM + 1
        if segValue == 85 :
           numeGM   = numeGM + 1
     elif GT_data[i,j,k] == 3 :
        denGTWM  = denGTWM + 1
        if segValue == 190 :
           numeWM   = numeWM + 1
     else :
        if GT_data[i,j,k] == 7 or GT_data[i,j,k] == 8 :
           denGTCSF = denGTCSF + 1
           if segValue == 255 :
              numeCSF = numeCSF + 1


DICE_GM  = 2.0*numeGM  / (denSMGM  + denGTGM)
DICE_WM  = 2.0*numeWM  / (denSMWM  + denGTWM)
DICE_CSF = 2.0*numeCSF / (denSMCSF + denGTCSF)

print "DICE GM = {}".format(DICE_GM)
print "DICE UWM = {}".format(DICE_WM)
print "DICE CSF = {}".format(DICE_CSF)


#   save results

SegMapVolume = nib.Nifti1Image(SegMap,affineT2CS)
nib.save(SegMapVolume,t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/SegMapVolume.nii.gz")

