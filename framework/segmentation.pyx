cimport cython
import numpy as np
cimport numpy as np
import scipy as sp
import math
import time
import sys
import matplotlib.pyplot as plt
import operator
import preprocessing
#import nibabel as nib
from PIL import Image
from skimage.morphology import ball,opening,closing,dilation,erosion,skeletonize #,watershed
#from scipy.ndimage.measurements import watershed_ift
from scipy import ndimage
from fast_morph import (SequencialSphereDilation,
                                  create_sphere,
                                  get_list,
                                  get_subsphere_lists,
                                  isotropic_erosion,
                                  isotropic_dilation)
from sklearn import cluster
from random import sample



##----------------------------------------------------------------------------------
##----------------------------------------------------------------------------------

#
# Function to get Largest Connected Components (LCCs).
# Input:  3D numpy array data where the LCCs are desired to be found, 3D
#         numpy array data_in_out that will be changed based on desired
#         LCCs for each case, integer case that specifies the LCCs case
#         to be computed (from 1 to 4), integer value v to which it is
#         desired data_in_out elements to be set when a desired
#         component is found or it also serves as index depending on
#         the case parameter and bounding box integers minI,minJ,minK,
#         maxI,maxJ,maxK that contain the bounding box of the
#         region of interest over the data array.
# Output: 3D numpy array data_in_out updated based on the computation
#         of the LCCs for each case.
#@cython.boundscheck(False)
def LCCs(np.ndarray[np.int64_t, ndim=3] data, np.ndarray[np.int64_t, ndim=3] data_in_out, \
         int case, int v, int minI, int minJ, int minK, \
         int maxI, int maxJ, int maxK) :
   cdef int i
   cdef int j
   cdef int k
   cdef int i_nc
   cdef int labelId
   cdef int firstLargest
   cdef int firstLargestId
   cdef int secondLargest
   cdef int secondLargestId
   #    uses default cross-shaped structuring element (for connectivity)
   labeledData,numcomponents = sp.ndimage.measurements.label(data)
   
   if numcomponents <= 1 and (case==1 or case==4):
      if case == 1 :
         return labeledData*255
      else:
         return labeledData
   
   componentLabels = np.array(np.zeros((numcomponents,2)),dtype=np.int64)
   for i_nc in range(0,numcomponents):
      labelId = i_nc + 1
      componentLabels[i_nc,0] = sum(sum(sum(labeledData==labelId)))
      componentLabels[i_nc,1] = labelId
   #componentLabels.sort(key=lambda tup: tup[0],reverse=True)
   firstLargest = 0
   secondLargest = 0
   firstLargestId = 0
   secondLargestId = 0
   for i in range(0,numcomponents):
      if componentLabels[i,0] > firstLargest :
         firstLargestId = componentLabels[i,1]
         firstLargest = componentLabels[i,0]
   for i in range(0,numcomponents):
      if componentLabels[i,0] <> firstLargest and componentLabels[i,0] > secondLargest :
         secondLargestId = componentLabels[i,1]
         secondLargest = componentLabels[i,0]
   
   if case == 1 :
      # keep only the two Largest Connected Components
      print "numcomp={}, firstL={}, secondL={}, firstLId={}, secondLId={}, maxCL={}".format(numcomponents,firstLargest,secondLargest,firstLargestId,secondLargestId,componentLabels.max())
      for i in range(minI,maxI+1):
        for j in range(minJ,maxJ+1):
          for k in range(minK,maxK+1):
            if labeledData[i,j,k] == firstLargestId or (secondLargest>0.25*firstLargest and labeledData[i,j,k] == secondLargestId) :
              data_in_out[i,j,k] = v #255
            else :
              data_in_out[i,j,k] = 0
   elif case == 2 :
      #     keep only Largest Connected Component
      for i in range(minI,maxI+1):
        for j in range(minJ,maxJ+1):
          for k in range(minK,maxK+1):
             if labeledData[i,j,k] <> firstLargestId :
                data_in_out[i,j,k] = v #255
   elif case == 3 :
      #     keep only Largest CSF Connected Component (together with background)
      #     and label the rest as UWM. Then keep only the two Largest Connected
      #     Components of UWM and label the rest as CSF. This gets rid of the
      #     misclassified CSF voxels at the CGM-UWM interface.
      for i in range(minI,maxI+1):
        for j in range(minJ,maxJ+1):
          for k in range(minK,maxK+1):
             if data_in_out[i,j,k] == 255 and labeledData[i,j,k] <> firstLargestId :
                data_in_out[i,j,k] = v #190
   else : #assumes case == 4
      # keep only the two Largest Connected Components
      for i in range(minI,maxI+1):
        for j in range(minJ,maxJ+1):
          for k in range(minK,maxK+1):
            if data_in_out[i,j,k] == 190 :
               if not (labeledData[i,j,k] == firstLargestId or (secondLargest>0.25*firstLargest and labeledData[i,j,k] == secondLargestId)) :
                  data_in_out[i,j,k] = v #255
   
   return data_in_out

#
# Function to filter mislabeled CSF inside UWM near border with CGM.
# Input:  3D numpy array Maskdata with the ICE mask, 3D
#         integer numpy array SegMap with the preliminar segmentation map
#         to be filtered, 3D double numpy array A_data with the desired
#         probability atlas in range [0,255] and bounding box
#         integers minI,minJ,minK,maxI,maxJ,maxK that contain the
#         bounding box of the region of interest over the data array.
# Output: 3D numpy array SegMap updated with the filtering.
@cython.boundscheck(False)
def filterCSF_in_UWMBorder(np.ndarray[np.double_t, ndim=3] Maskdata, np.ndarray[np.int64_t, ndim=3] SegMap, \
                            np.ndarray[np.double_t, ndim=3] A_data, \
                            int minI, int minJ, int minK, \
                            int maxI, int maxJ, int maxK, \
                            int ss4_num_ne) :
   cdef int rep
   cdef int i
   cdef int j
   cdef int k
   for rep in range(0,30):
    changesCount = 0
    for i in range(minI,maxI+1):
     for j in range(minJ,maxJ+1):
      for k in range(minK,maxK+1):
        val = Maskdata[i,j,k]
        if val >0 :
           vN1 = SegMap[i-1,j,k]
           vN2 = SegMap[i+1,j,k]
           vN3 = SegMap[i,j-1,k]
           vN4 = SegMap[i,j+1,k]
           vN5 = SegMap[i,j,k-1]
           vN6 = SegMap[i,j,k+1]
           if SegMap[i,j,k] == 255 and A_data[i,j,k] >= 26 : #ACSF_data[i,j,k] >= 26 :
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
             if count1 >= ss4_num_ne and wmN :
                SegMap[i,j,k] = 190
                changesCount = changesCount + 1
    print "uwm changes={}".format(changesCount)
    if changesCount == 0 :
       break
    
    return SegMap



##-----------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------

def CGMEnhancing(np.ndarray[np.double_t, ndim=3] dataT2, np.ndarray[np.double_t, ndim=3] dataT1, \
         np.ndarray[np.int64_t, ndim=3] TrainingSegMap, \
         str output_dir, int minI, int minJ, int minK, int maxI, int maxJ, int maxK, \
         int nSlice, int difSlice, int dim1, int dim2, int dim3, \
         double vox_spacing, \
         double par_gamma, double par_priorthresh, int useT1) :
   cdef int i,j,k,i1,j1,count1,count2,count3,wsizecgm,dil_rad
   cdef double cgm_thickness = 1.5  # typical minimum reported thickness for CGM in milimeters
   cdef double filterTol,xTol
   cdef double val,val1,val2
   cdef double S,maxS
   cdef double Lamb1
   cdef double Lamb2
   cdef double Lamb3
   cdef double Rb,maxRb
   cdef double Ra,maxRa
   cdef double Vline
   cdef double Vplate
   dil_rad  = int(0.5*(cgm_thickness / vox_spacing)) # <- floor operation
   wsizecgm = dil_rad*2 + 1
   print "CGM thickness parameters: dil_rad = {}, wsizecgm = {}".format(dil_rad,wsizecgm)
   
   
   # perform dilation of the registered training segmentation map on the CGM zone
   struct_elem = ball(dil_rad)
   CGMMap = np.array(np.zeros((dim1,dim2,dim3)),dtype=np.int64)
   CGMMap[ TrainingSegMap == 1 ] = 1
   CGMMap = dilation(CGMMap,struct_elem)
   Im = Image.fromarray(np.uint8(CGMMap[:,:,nSlice]*255))
   Im.save(output_dir+'SEGM_trainingCGMDilatedMap.png')
   Im = Image.fromarray(np.uint8(CGMMap[:,:,nSlice+difSlice]*255))
   Im.save(output_dir+'SEGM_trainingCGMDilatedMap_p'+str(difSlice)+'.png')
   Im = Image.fromarray(np.uint8(CGMMap[:,:,nSlice- difSlice]*255))
   Im.save(output_dir+'SEGM_trainingCGMDilatedMap_m'+str(difSlice)+'.png')
   
   filterTol = 1.0e-08
   xTol      = - math.log(filterTol)
   # compute Hessian of T2 subject image
   grads  = np.gradient(dataT2)
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
   Iyy    = gradsy[1]
   Iyz    = gradsy[2]
   del gradsy
   gradsz = np.gradient(Iz)
   Izz    = gradsz[2]
   del gradsz
   del Ix
   del Iy
   del Iz
   #   compute maximum of S, Ra and Rb
   maxRb  = 0
   maxRa  = 0
   maxS   = 0
   for i in range(minI,maxI+1):
    for j in range(minJ,maxJ+1):
     for k in range(minK,maxK+1):
       val = dataT2[i,j,k]
       val1 = CGMMap[i,j,k]
       if val > 0 and val1 > 0 :
             # compute Hessian eigenvalues
             H = np.array([[Ixx[i,j,k], Ixy[i,j,k], Ixz[i,j,k]], [Ixy[i,j,k], Iyy[i,j,k], Iyz[i,j,k]], [Ixz[i,j,k], Iyz[i,j,k], Izz[i,j,k]]],dtype=float)
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
             if Rb > maxRb :
                maxRb = Rb
             if Ra > maxRa :
                maxRa = Ra
   print "Maximum value of S = {}".format(maxS)
   print "Maximum of: Rb={}, Ra={}".format(maxRb,maxRa)
   
   #     Apply line-like and plate-like filters based on eigenvalues
   #     of the Hessian matrix on the T2 image
   CGMHessianPrior = np.zeros((dim1,dim2,dim3),dtype=np.double)
   for i in range(minI,maxI+1):
    for j in range(minJ,maxJ+1):
     for k in range(minK,maxK+1):
       val = dataT2[i,j,k]
       val1 = CGMMap[i,j,k]
       if val > 0 and val1 > 0 :
             # compute Hessian eigenvalues
             H = np.array([[Ixx[i,j,k], Ixy[i,j,k], Ixz[i,j,k]], [Ixy[i,j,k], Iyy[i,j,k], Iyz[i,j,k]], [Ixz[i,j,k], Iyz[i,j,k], Izz[i,j,k]]],dtype=float)
             # get ordered eigenvalues
             egval, _ = np.linalg.eig(H)
             egvalList = [ (abs(egval[0]),egval[0]), (abs(egval[1]),egval[1]), (abs(egval[2]),egval[2]) ]
             egvalList.sort(key=lambda tup: tup[0])
             Lamb1 = egvalList[0][1]
             Lamb2 = egvalList[1][1]
             Lamb3 = egvalList[2][1]
             # apply line-like filter and plate-like filter to current voxel
             S = math.sqrt(Lamb1**2 + Lamb2**2 + Lamb3**2)
             Rb = abs(Lamb1) / math.sqrt(abs(Lamb2*Lamb3))
             Ra = abs(Lamb2) / abs(Lamb3)
             val2 = math.exp(-0.5*(Rb**2)/((par_gamma*maxRb)**2)) * ( 1.0-math.exp(-0.5*(S**2)/((par_gamma*maxS)**2)) )
             Vline = 0
             if Lamb2 > 0 and Lamb3 > 0 :
                Vline = 1.0 - math.exp( -0.5*(Ra**2)/((par_gamma*maxRa)**2) )
                Vline = Vline * val2
             Vplate = 0
             if Lamb3 > 0 :
                Vplate = math.exp( -0.5*(Ra**2)/((par_gamma*maxRa)**2) )
                Vplate = Vplate * val2
             CGMHessianPrior[i,j,k] = max(max(Vline,Vplate),filterTol)
             CGMHessianPrior[i,j,k] = max((255.0/xTol)*(math.log(CGMHessianPrior[i,j,k]) + xTol),0)
   
   print "Hessian filter found values:"
   print "   maxCGMPriorT2={}, minCGMPriorT2={}".format(CGMHessianPrior.max(),CGMHessianPrior.min())
   
   Im = Image.fromarray(np.uint8(CGMHessianPrior[:,:,nSlice]))
   Im.save(output_dir+'SEGM_CGMHessianPriorT2.png')
   Im = Image.fromarray(np.uint8(CGMHessianPrior[:,:,nSlice+difSlice]))
   Im.save(output_dir+'SEGM_CGMHessianPriorT2_p'+str(difSlice)+'.png')
   Im = Image.fromarray(np.uint8(CGMHessianPrior[:,:,nSlice-difSlice]))
   Im.save(output_dir+'SEGM_CGMHessianPriorT2_m'+str(difSlice)+'.png')
   
   if useT1 == 1 :
      # compute Hessian of T1 subject image
      grads  = np.gradient(dataT1)
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
      Iyy    = gradsy[1]
      Iyz    = gradsy[2]
      del gradsy
      gradsz = np.gradient(Iz)
      Izz    = gradsz[2]
      del gradsz
      del Ix
      del Iy
      del Iz
      #   compute maximum of S, Ra and Rb
      maxS = 0
      maxRa = 0
      maxRb = 0
      for i in range(minI,maxI+1):
       for j in range(minJ,maxJ+1):
        for k in range(minK,maxK+1):
          val = dataT2[i,j,k]
          val1 = CGMMap[i,j,k]
          if val > 0 and val1 > 0 :
                # compute Hessian eigenvalues
                H = np.array([[Ixx[i,j,k], Ixy[i,j,k], Ixz[i,j,k]], [Ixy[i,j,k], Iyy[i,j,k], Iyz[i,j,k]], [Ixz[i,j,k], Iyz[i,j,k], Izz[i,j,k]]],dtype=float)
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
                if Rb > maxRb :
                   maxRb = Rb
                if Ra > maxRa :
                   maxRa = Ra
      print "Maximum value of S = {}".format(maxS)
      print "Maximum of: Rb={}, Ra={}".format(maxRb,maxRa)
      
      #     Apply line-like and plate-like filters based on eigenvalues
      #     of the Hessian matrix on the T1 image
      CGMHessianPrior2 = np.zeros((dim1,dim2,dim3),dtype=np.double)
      for i in range(minI,maxI+1):
       for j in range(minJ,maxJ+1):
        for k in range(minK,maxK+1):
          val = dataT2[i,j,k]
          val1 = CGMMap[i,j,k]
          if val > 0 and val1 > 0 :
                # compute Hessian eigenvalues
                H = np.array([[Ixx[i,j,k], Ixy[i,j,k], Ixz[i,j,k]], [Ixy[i,j,k], Iyy[i,j,k], Iyz[i,j,k]], [Ixz[i,j,k], Iyz[i,j,k], Izz[i,j,k]]],dtype=float)
                # get ordered eigenvalues
                egval, _ = np.linalg.eig(H)
                egvalList = [ (abs(egval[0]),egval[0]), (abs(egval[1]),egval[1]), (abs(egval[2]),egval[2]) ]
                egvalList.sort(key=lambda tup: tup[0])
                Lamb1 = egvalList[0][1]
                Lamb2 = egvalList[1][1]
                Lamb3 = egvalList[2][1]
                # apply line-like filter and plate-like filter to current voxel
                S = math.sqrt(Lamb1**2 + Lamb2**2 + Lamb3**2)
                Rb = abs(Lamb1) / math.sqrt(abs(Lamb2*Lamb3))
                Ra = abs(Lamb2) / abs(Lamb3)
                val2 = math.exp(-0.5*(Rb**2)/((par_gamma*maxRb)**2)) * ( 1.0-math.exp(-0.5*(S**2)/((par_gamma*maxS)**2)) )
                Vline = 0
                if Lamb2 < 0 and Lamb3 < 0 :
                   Vline = 1.0 - math.exp( -0.5*(Ra**2)/((par_gamma*maxRa)**2) )
                   Vline = Vline * val2
                Vplate = 0
                if Lamb3 < 0 :
                   Vplate = math.exp( -0.5*(Ra**2)/((par_gamma*maxRa)**2) )
                   Vplate = Vplate * val2
                CGMHessianPrior2[i,j,k] = max(max(Vline,Vplate),filterTol)
                CGMHessianPrior2[i,j,k] = max((255.0/xTol)*(math.log(CGMHessianPrior2[i,j,k]) + xTol),0)
      
      print "Hessian filter found values:"
      print "   maxCGMPriorT1={}, minCGMPriorT1={}".format(CGMHessianPrior2.max(),CGMHessianPrior2.min())
      CGMHessianPrior  = (CGMHessianPrior + CGMHessianPrior2) / 2.0
      Im = Image.fromarray(np.uint8(CGMHessianPrior2[:,:,nSlice]))
      Im.save(output_dir+'SEGM_CGMHessianPriorT1.png')
      Im = Image.fromarray(np.uint8(CGMHessianPrior2[:,:,nSlice+difSlice]))
      Im.save(output_dir+'SEGM_CGMHessianPriorT1_p'+str(difSlice)+'.png')
      Im = Image.fromarray(np.uint8(CGMHessianPrior2[:,:,nSlice-difSlice]))
      Im.save(output_dir+'SEGM_CGMHessianPriorT1_m'+str(difSlice)+'.png')
      Im = Image.fromarray(np.uint8(CGMHessianPrior[:,:,nSlice]))
      Im.save(output_dir+'SEGM_CGMHessianPriorT1T2.png')
      Im = Image.fromarray(np.uint8(CGMHessianPrior[:,:,nSlice+difSlice]))
      Im.save(output_dir+'SEGM_CGMHessianPriorT1T2_p'+str(difSlice)+'.png')
      Im = Image.fromarray(np.uint8(CGMHessianPrior[:,:,nSlice-difSlice]))
      Im.save(output_dir+'SEGM_CGMHessianPriorT1T2_m'+str(difSlice)+'.png')
      del CGMHessianPrior2
   
   
   # compute modified prior cgm segmentation map based on the registered
   # segmentation map from the training subject. This function emphasizes
   # the isotropic regions of the cgm segmentation map to compensate this
   # type of regions on the hessian filtered function.
   CGMTrainingPrior = np.zeros((dim1,dim2,dim3),dtype=np.double)
   for i in range(minI,maxI+1):
    for j in range(minJ,maxJ+1):
     for k in range(minK,maxK+1):
       val = dataT2[i,j,k]
       val1 = TrainingSegMap[i,j,k]
       if val > 0 and val1 == 1 :
          count1 = 0
          for i1 in range(-int(wsizecgm/2),int(wsizecgm/2)+1,1):
           for j1 in range(-int(wsizecgm/2),int(wsizecgm/2)+1,1):
              if TrainingSegMap[i+i1,j+j1,k] == 1 :
                 count1 = count1 + 1
          CGMTrainingPrior[i,j,k] = 255.0 * np.double(count1) / np.double(wsizecgm**2)
   # combine cgm prior hessian and training map information to produce a CGM
   # seed (saved in CGMMap)
   SegMap    = np.array(np.zeros((dim1,dim2,dim3)),dtype=np.int64)
   for i in range(minI,maxI+1):
    for j in range(minJ,maxJ+1):
     for k in range(minK,maxK+1):
       val = dataT2[i,j,k]
       if val > 0 :
          val1 = (0.5*CGMHessianPrior[i,j,k] + 0.5*CGMTrainingPrior[i,j,k])
          if val1 > par_priorthresh :
             SegMap[i,j,k] = 80 # <-- code class for voxels candidates to becoming cgm
          else :
             CGMMap[i,j,k] = 0
   # region growing algorithm that starts with CGMMap as seed and adds by coronal slice
   # first order neighbouring candidates from regions with label SegMap == 80
   for k in range(minK,maxK+1):
    #CGMCandidates = []
    while True :
       CGMCandidates = []
       for i in range(minI,maxI+1):
        for j in range(minJ,maxJ+1):
           if CGMMap[i,j,k] == 1 :
              if CGMMap[i+1,j,k] == 0 and SegMap[i+1,j,k] == 80 :
                 CGMCandidates.append( (i+1,j) )
              if CGMMap[i-1,j,k] == 0 and SegMap[i-1,j,k] == 80 :
                 CGMCandidates.append( (i-1,j) )
              if CGMMap[i,j+1,k] == 0 and SegMap[i,j+1,k] == 80 :
                 CGMCandidates.append( (i,j+1) )
              if CGMMap[i,j-1,k] == 0 and SegMap[i,j-1,k] == 80 :
                 CGMCandidates.append( (i,j-1) )
       if len(CGMCandidates) == 0 :
          break
       else :
          for i in range(0,len(CGMCandidates)):
           i1 = CGMCandidates[i][0]
           j1 = CGMCandidates[i][1]
           CGMMap[i1,j1,k] = 1
           if TrainingSegMap[i1,j1,k] == 2 :
              SegMap[i1,j1,k] = 50 # code class for sgm
           elif (TrainingSegMap[i1,j1,k] == 7 or TrainingSegMap[i1,j1,k] == 8) :
              SegMap[i1,j1,k] = 255 # code class for csf
           else :
              count1 = 0
              count2 = 0
              count3 = 0
              if (TrainingSegMap[i1+1,j1,k] == 7 or TrainingSegMap[i1+1,j1,k] == 8) :
                 count1 = count1 + 1
              else:
                 if TrainingSegMap[i1+1,j1,k] == 3 :
                    count2 = count2 + 1
                 else :
                    if TrainingSegMap[i1+1,j1,k] == 1 :
                       count3 = count3 + 1
              if (TrainingSegMap[i1-1,j1,k] == 7 or TrainingSegMap[i1-1,j1,k] == 8) :
                 count1 = count1 + 1
              else:
                 if TrainingSegMap[i1-1,j1,k] == 3 :
                    count2 = count2 + 1
                 else :
                    if TrainingSegMap[i1-1,j1,k] == 1 :
                       count3 = count3 + 1
              if (TrainingSegMap[i1,j1+1,k] == 7 or TrainingSegMap[i1,j1+1,k] == 8) :
                 count1 = count1 + 1
              else:
                 if TrainingSegMap[i1,j1+1,k] == 3 :
                    count2 = count2 + 1
                 else :
                    if TrainingSegMap[i1,j1+1,k] == 1 :
                       count3 = count3 + 1
              if (TrainingSegMap[i1,j1-1,k] == 7 or TrainingSegMap[i1,j1-1,k] == 8) :
                 count1 = count1 + 1
              else:
                 if TrainingSegMap[i1,j1-1,k] == 3 :
                    count2 = count2 + 1
                 else :
                    if TrainingSegMap[i1,j1-1,k] == 1 :
                       count3 = count3 + 1
              if count1 > count2 :
                 SegMap[i1,j1,k] = 255
              else :
                 if count2 > 0 :
                    SegMap[i1,j1,k] = 190 # code class for uwm
                 else:
                    if count3 > 0 :
                       SegMap[i1,j1,k] = 85
   
   Im = Image.fromarray(np.uint8(CGMMap[:,:,nSlice]*255))
   Im.save(output_dir+'SEGM_CGMMap.png')
   Im = Image.fromarray(np.uint8(CGMMap[:,:,nSlice+difSlice]*255))
   Im.save(output_dir+'SEGM_CGMMap_p'+str(difSlice)+'.png')
   Im = Image.fromarray(np.uint8(CGMMap[:,:,nSlice-difSlice]*255))
   Im.save(output_dir+'SEGM_CGMMap_m'+str(difSlice)+'.png')
   #     Finish labeling the rest of the volume.
   for i in range(minI,maxI+1):
    for j in range(minJ,maxJ+1):
     for k in range(minK,maxK+1):
        val = dataT2[i,j,k]
        if val > 0 :
           if CGMMap[i,j,k] == 1 :
              SegMap[i,j,k] = 85 # code class for cgm
           else :
              if TrainingSegMap[i,j,k] == 2 :
                 SegMap[i,j,k] = 50 # code class for sgm
              elif (TrainingSegMap[i,j,k] == 7 or TrainingSegMap[i,j,k] == 8) :
                 SegMap[i,j,k] = 255 # code class for csf
              else :
                 count1 = 0
                 count2 = 0
                 count3 = 0
                 if (TrainingSegMap[i+1,j,k] == 7 or TrainingSegMap[i+1,j,k] == 8) :
                    count1 = count1 + 1
                 else:
                    if TrainingSegMap[i+1,j,k] == 3 :
                       count2 = count2 + 1
                    else :
                       if TrainingSegMap[i+1,j,k] == 1 :
                          count3 = count3 + 1
                 if (TrainingSegMap[i-1,j,k] == 7 or TrainingSegMap[i-1,j,k] == 8) :
                    count1 = count1 + 1
                 else:
                    if TrainingSegMap[i-1,j,k] == 3 :
                       count2 = count2 + 1
                    else :
                       if TrainingSegMap[i-1,j,k] == 1 :
                          count3 = count3 + 1
                 if (TrainingSegMap[i,j+1,k] == 7 or TrainingSegMap[i,j+1,k] == 8) :
                    count1 = count1 + 1
                 else:
                    if TrainingSegMap[i,j+1,k] == 3 :
                       count2 = count2 + 1
                    else :
                       if TrainingSegMap[i,j+1,k] == 1 :
                          count3 = count3 + 1
                 if (TrainingSegMap[i,j-1,k] == 7 or TrainingSegMap[i,j-1,k] == 8) :
                    count1 = count1 + 1
                 else:
                    if TrainingSegMap[i,j-1,k] == 3 :
                       count2 = count2 + 1
                    else :
                       if TrainingSegMap[i,j-1,k] == 1 :
                          count3 = count3 + 1
                 if count1 > count2 :
                    SegMap[i,j,k] = 255
                 else :
                    if count2 > 0 :
                       SegMap[i,j,k] = 190 # code class for uwm
                    else :
                       if count3 > 0 :
                          SegMap[i,j,k] = 85
   
   #   Remove unconnected CSF inside UWM
   UWMBinary = np.array(np.ones((dim1,dim2,dim3)),dtype=np.int)
   UWMBinary[ SegMap == 255 ] = 0
   
   SegMap = LCCs(np.array(UWMBinary,dtype=np.int64), np.array(SegMap,dtype=np.int64), \
                 2,255, minI,minJ,minK,maxI,maxJ,maxK)
   
   print "UWM Corrected"
   del UWMBinary
   
   Im = Image.fromarray(np.uint8(SegMap[:,:,nSlice]))
   Im.save(output_dir+'SEGM_SegmCGMEnhanced.png')
   Im = Image.fromarray(np.uint8(SegMap[:,:,nSlice+difSlice]))
   Im.save(output_dir+'SEGM_SegmCGMEnhanced_p'+str(difSlice)+'.png')
   Im = Image.fromarray(np.uint8(SegMap[:,:,nSlice-difSlice]))
   Im.save(output_dir+'SEGM_SegmCGMEnhanced_m'+str(difSlice)+'.png')
   
   return SegMap


#
# SGM zone segmentation enhancing based on the EM algorithm (Makropoulos, 2015).
# Input:  dataT2 numpy array with the 3D data of the subject T2 MRI image,
#         preliminar 3D numpy array segmentation map SegMap, arrays AWM_data,
#         ADGM_data,ACSF_data,A50_data corresponding to the WM,DGM and CSF atlas
#         probability maps plus the atlas array A50_data with the probability maps
#         (in the intensity range of [0,255]) of 50 regions of the infant brain,
#         physical voxel spacing scalar voxel_spacing, output directory name
#         output_dir for printing temporal slices of the process, bounding box
#         coordinates minI,minJ,minK,maxI,maxJ,maxK of the ICE corresponding to the
#         dataT2 array for each axis, base slice nSlice to show during the process,
#         slice gap difSlice whose corresponding slice should also be shown around
#         nSlice and dimensions of the data array dim1, dim2, dim3.
# Output: numpy array segmentation map SegMap with the improved
#         segmentation of the tissues (UWM = 190, CGM = 85, SGM = 50, CSF = 255).
@cython.boundscheck(False)
def SGMSegmEnhancing(np.ndarray[np.double_t, ndim=3] dataT2, np.ndarray[np.double_t, ndim=3] dataT1, \
         np.ndarray[np.int64_t, ndim=3] SegMap, np.ndarray[np.int64_t, ndim=3] ASegMap, \
         np.ndarray[np.double_t, ndim=3] AT2_data, np.ndarray[np.double_t, ndim=3] AT1_data, \
         str output_dir, int minI, int minJ, int minK, int maxI, int maxJ, int maxK, \
         int dim3, int dim3GT, \
         int nSlice, int difSlice, \
         int nln, int nnln, int r, int ps, int ra, int useT1) :
   cdef int i,j,k,i1,i2,i3,j1,k1,k_gt
   cdef double val1,val2
   cdef int csf,uwm,sgm,cgm
   cdef double nume,denom,dotProd
   cdef int s
   cdef int s_2,r_2,ps_2,width,ind
   #ra = 0
   s = 3
   #ps = 3
   #r = 5
   #nln = 4
   #nnln = 4
   LocalIndexSet = []
   NonLocalIndexSet = []
   s_2 = int(s/2)
   r_2 = int(r/2)
   ps_2 = int(ps/2)
   width = r_2 - s_2
   
   ind = 0
   for j in range(0,width):
    for i in range(-r_2,r_2+1):
       NonLocalIndexSet.append((i,r_2 - j))
       ind = ind + 1
   for j in range(width,r-width):
    for i in range(-r_2,-r_2+width):
       NonLocalIndexSet.append((i,r_2 - j))
       ind = ind + 1
    for i in range(r_2-(width-1),r_2+1):
       NonLocalIndexSet.append((i,r_2 - j))
       ind = ind + 1
   for j in range(width-1,-1,-1):
    for i in range(-r_2,r_2+1):
       NonLocalIndexSet.append((i,-r_2+j))
       ind = ind + 1
   
   if nln == 4 :
      LocalIndexSet.append((-1,0))
      LocalIndexSet.append((0,1))
      LocalIndexSet.append((1,0))
      LocalIndexSet.append((0,-1))
      NonLocalIndexSet.append((-1,1))
      NonLocalIndexSet.append((1,1))
      NonLocalIndexSet.append((1,-1))
      NonLocalIndexSet.append((-1,-1))
   else : # nln == 8 neighbours
      if nln == 0 and nnln == 0 :
         LocalIndexSet = []
         LocalIndexSet.append((0,0))
         NonLocalIndexSet = []
      else :
         LocalIndexSet.append((-1,0))
         LocalIndexSet.append((-1,1))
         LocalIndexSet.append((0,1))
         LocalIndexSet.append((1,1))
         LocalIndexSet.append((1,0))
         LocalIndexSet.append((1,-1))
         LocalIndexSet.append((0,-1))
         LocalIndexSet.append((-1,-1))
   NonLocalIndexSet_Random = NonLocalIndexSet
   
   # Obtain improved SGM information
   for k_gt in range(0,dim3GT):
    k = int(round(float(dim3*k_gt)/float(dim3GT)))
    if k >= minK and k <= maxK :
       #for k in range(minK,maxK+1):
       for i in range(minI+s_2,maxI+1-s_2):
        for j in range(minJ+s_2,maxJ+1-s_2):
         if dataT2[i,j,k] > 0 :
          cgm = 0
          sgm = 0
          uwm = 0
          csf = 0
          # Local window neighbourhood
          for i1 in range(0,len(LocalIndexSet)):
             if ASegMap[i+LocalIndexSet[i1][0],j+LocalIndexSet[i1][1],k] == 1 :
                cgm = cgm + 1
             elif ASegMap[i+LocalIndexSet[i1][0],j+LocalIndexSet[i1][1],k] == 2 :
                sgm = sgm + 1
             elif ASegMap[i+LocalIndexSet[i1][0],j+LocalIndexSet[i1][1],k] == 3 :
                uwm = uwm + 1
             else:
                if ASegMap[i+LocalIndexSet[i1][0],j+LocalIndexSet[i1][1],k] == 7 or \
                   ASegMap[i+LocalIndexSet[i1][0],j+LocalIndexSet[i1][1],k] == 8 :
                   csf = csf + 1
          # Non-local neighbourhood
          if ra > 0 :
             NonLocalIndexSet = sample(NonLocalIndexSet_Random,int(0.5*len(NonLocalIndexSet_Random)))
          if len(NonLocalIndexSet) > 0 :
                dotProds_NLIndexSet = []
                val1 = 0
                val2 = 0
                dotProd = 0
                for i1 in range(0,len(NonLocalIndexSet)):
                   # compute similarity and get most similar nnln non-local neighbours
                   for i2 in range(-ps_2,ps_2+1):
                    for i3 in range(-ps_2,ps_2+1):
                       val1 = dataT2[i+NonLocalIndexSet[i1][0]+i2,j+NonLocalIndexSet[i1][1]+i3,k]
                       val2 = AT2_data[i+NonLocalIndexSet[i1][0]+i2,j+NonLocalIndexSet[i1][1]+i3,k]
                       dotProd = dotProd + val1*val2
                       if useT1 :
                          val1 = dataT1[i+NonLocalIndexSet[i1][0]+i2,j+NonLocalIndexSet[i1][1]+i3,k]
                          val2 = AT1_data[i+NonLocalIndexSet[i1][0]+i2,j+NonLocalIndexSet[i1][1]+i3,k]
                          dotProd = dotProd + val1*val2
                   dotProds_NLIndexSet.append((dotProd,NonLocalIndexSet[i1][0],NonLocalIndexSet[i1][1]))
                dotProds_NLIndexSet.sort(key=lambda tup: tup[0], reverse=True)
                for i1 in range(0,nnln):
                   if ASegMap[i+dotProds_NLIndexSet[i1][1],j+dotProds_NLIndexSet[i1][2],k] == 1 :
                      cgm = cgm + 1
                   elif ASegMap[i+dotProds_NLIndexSet[i1][1],j+dotProds_NLIndexSet[i1][2],k] == 2 :
                      sgm = sgm + 1
                   elif ASegMap[i+dotProds_NLIndexSet[i1][1],j+dotProds_NLIndexSet[i1][2],k] == 3 :
                      uwm = uwm + 1
                   else:
                      if ASegMap[i+dotProds_NLIndexSet[i1][1],j+dotProds_NLIndexSet[i1][2],k] == 7 or \
                         ASegMap[i+dotProds_NLIndexSet[i1][1],j+dotProds_NLIndexSet[i1][2],k] == 8 :
                         csf = csf + 1
          
          # assign new segmentation based on max mode value (most frequent label)
          i1 = i-minI
          j1 = j-minJ
          k1 = k-minK
          val1 = max(cgm,max(sgm,max(uwm,csf)))
          if val1 > 0 :
             if ASegMap[i,j,k] == 2 :
                if val1 == np.double(cgm) :
                   SegMap[i,j,k] = 85
                elif val1 == np.double(sgm) :
                   SegMap[i,j,k] = 50
                elif val1 == np.double(csf) :
                   SegMap[i,j,k] = 255
                else :
                   SegMap[i,j,k] = 190
             else :
                if val1 == np.double(sgm) :
                   SegMap[i,j,k] = 50
   Im = Image.fromarray(np.uint8(SegMap[:,:,nSlice]))
   Im.save(output_dir+'SEGM_SegmSGMEnhanced.png')
   Im = Image.fromarray(np.uint8(SegMap[:,:,nSlice+difSlice]))
   Im.save(output_dir+'SEGM_SegmSGMEnhanced_p'+str(difSlice)+'.png')
   Im = Image.fromarray(np.uint8(SegMap[:,:,nSlice-difSlice]))
   Im.save(output_dir+'SEGM_SegmSGMEnhanced_m'+str(difSlice)+'.png')
   
   return SegMap


##-----------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------

#
# Fuzzy C-Means with Non-Local Spatial information (FCM_NLS) for gray level 3D images.
# Input:  3D numpy array data with the data to segment, 1D numpy array V
#         with the initial centers of each class of shape (K,), 1D numpy array labels
#         with the integer labels to set to each class in the same order as the cluster
#         centers array V, K integer number of desired classes, r integer window size
#         for non-local information, s integer local window size, h parameter for the
#         Gaussian function, beta parameter for the NLS term, m parameter for the FCM
#         term, tole tolerance level for ending iterations and MAX_ITE max number of
#         iterations.
# Output: 3D numpy array SegMap with the segmentation.
@cython.boundscheck(False)
def PFCM(np.ndarray[np.double_t, ndim=3] data, np.ndarray[np.double_t, ndim=1] V, \
             np.ndarray[np.double_t, ndim=4] U, np.ndarray[np.int64_t, ndim=1] labels, \
             #int K, int r, int s, double h, double beta, double m, double tole, int MAX_ITE) :
             int K, double eta, double m, double tole, int MAX_ITE) :
   cdef int i,j,k,k1,k2,it
   cdef int dim1,dim2,dim3
   cdef double nume,denom,sum1,val,w,d
   dim1 = data.shape[0]
   dim2 = data.shape[1]
   dim3 = data.shape[2]
   print "data.shape=({},{},{}), U.shape=({},{},{})".format(data.shape[0],data.shape[1],data.shape[2], \
                        U.shape[0],U.shape[1],U.shape[2])
   W         = np.array(np.zeros((dim1,dim2,dim3,K)),dtype=np.double)
   #U         = np.array(np.zeros((dim1,dim2,dim3,K)),dtype=np.double)
   T         = np.array(np.zeros((dim1,dim2,dim3,K)),dtype=np.double)
   V_next    = np.array(V,dtype=np.double)
   V         = V - np.array(np.ones(K),dtype=np.double)
   mu        = np.array(np.zeros(K),dtype=np.double)
   n         = np.array(np.zeros(K),dtype=np.int64)
   sigma     = np.array(np.zeros(K),dtype=np.double)
   #  apply iterative algorithm
   it = 1
   while it <= MAX_ITE : #np.linalg.norm(V_next - V) >= tole and it <= MAX_ITE :
      V = V_next
      # update weights
      for k1 in range(0,K):
         mu[k1] = 0
         sigma[k1] = 0
         n[k1] = 0
      for i in range(0,dim1):
       for j in range(0,dim2):
        for k in range(0,dim3):
           if data[i,j,k] > 0 :
              k2  = 0
              val = U[i,j,k,k2]
              for k1 in range(1,K):
                if U[i,j,k,k1] > val :
                   val = U[i,j,k,k1]
                   k2 = k1
              mu[k2] = mu[k2] + data[i,j,k]
              n[k2]  = n[k2] + 1
      for k1 in range(0,K):
         mu[k1] = mu[k1] / np.double(n[k1])
      for i in range(0,dim1):
       for j in range(0,dim2):
        for k in range(0,dim3):
           if data[i,j,k] > 0 :
              k2  = 0
              val = U[i,j,k,k2]
              for k1 in range(1,K):
                if U[i,j,k,k1] > val :
                   val = U[i,j,k,k1]
                   k2 = k1
              sigma[k2] = sigma[k2] + (data[i,j,k]-mu[k2])**2
      for k1 in range(0,K):
         sigma[k1] = math.sqrt(sigma[k1] / np.double(n[k1]-1.0))
      for i in range(0,dim1):
       for j in range(0,dim2):
        for k in range(0,dim3):
           if data[i,j,k] > 0 :
              for k1 in range(0,K):
                 W[i,j,k,k1] = (1.0/(math.sqrt(2*math.pi)*sigma[k1])) * math.exp( -(data[i,j,k]-mu[k1])**2 / ( 2.0*(sigma[k1]**2) ) )
      
      # update U fuzzy membership functions
      for i in range(0,dim1):
       for j in range(0,dim2):
        for k in range(0,dim3):
          if data[i,j,k] > 0 :
             for k1 in range(0,K):
                denom = 0
                for k2 in range(0,K):
                   w     = W[i,j,k,k2]
                   d     = (data[i,j,k] - V[k2])**(2)
                   denom = denom + (  (w**m)*d + ((1.0-T[i,j,k,k2])**eta)*(w**eta)*d  ) ** (-1.0/(m-1))
                w           = W[i,j,k,k1]
                d           = (data[i,j,k] - V[k1])**(2)
                nume        = (w**m)*d + ((1.0-T[i,j,k,k1])**eta)*(w**eta)*d
                U[i,j,k,k1] =  ((1.0 / nume)**(m-1))  /  denom
      # update T possibilistic membership functions
      for i in range(0,dim1):
       for j in range(0,dim2):
        for k in range(0,dim3):
          if data[i,j,k] > 0 :
             for k1 in range(0,K):
                val = ( (U[i,j,k,k1]**m) * (W[i,j,k,k1]**(eta-m)) ) **(1.0/(eta-1.0))
                T[i,j,k,k1] = val / (1.0 + val)
      # update cluster centers
      for k1 in range(0,K):
        nume = 0
        denom = 0
        for i in range(0,dim1):
         for j in range(0,dim2):
          for k in range(0,dim3):
             if data[i,j,k] > 0 :
                nume = nume + (U[i,j,k,k1]**m + T[i,j,k,k1]**eta)*(W[i,j,k,k1]**m)*data[i,j,k]
                nume = nume + (U[i,j,k,k1]**m)*((1.0-T[i,j,k,k1])**eta)*(W[i,j,k,k1]**eta)*data[i,j,k]
                denom = denom + (U[i,j,k,k1]**m + T[i,j,k,k1]**eta)*(W[i,j,k,k1]**m)
                denom = denom + (U[i,j,k,k1]**m)*((1.0-T[i,j,k,k1])**eta)*(W[i,j,k,k1]**eta)
        V_next[k1] = nume / denom
      print "------PFCM, iteration {} done.".format(it)
      it = it + 1
   del T
   del W
   # final labeling according to membership functions
   SegMap  = np.array(np.zeros((dim1,dim2,dim3)),dtype=np.int64)
   for i in range(0,dim1):
    for j in range(0,dim2):
     for k in range(0,dim3):
        if data[i,j,k] > 0 :
           k2  = 0
           val = U[i,j,k,k2]
           for k1 in range(1,K):
             if U[i,j,k,k1] > val :
                val = U[i,j,k,k1]
                k2 = k1
           SegMap[i,j,k] = labels[k2]
   del U
   
   return SegMap
#
# Fuzzy C-Means with Non-Local Spatial information (FCM_NLS) for gray level 3D images.
# Input:  3D numpy array data with the data to segment, 1D numpy array V
#         with the initial centers of each class of shape (K,), 1D numpy array labels
#         with the integer labels to set to each class in the same order as the cluster
#         centers array V, K integer number of desired classes, r integer window size
#         for non-local information, s integer local window size, h parameter for the
#         Gaussian function, beta parameter for the NLS term, m parameter for the FCM
#         term, tole tolerance level for ending iterations and MAX_ITE max number of
#         iterations.
# Output: 3D numpy array SegMap with the segmentation.
@cython.boundscheck(False)
def FCM_NLS(np.ndarray[np.double_t, ndim=3] data, np.ndarray[np.double_t, ndim=1] V, \
             np.ndarray[np.int64_t, ndim=1] labels, \
             int K, int r, int s, double h, double beta, double m, double tole, int MAX_ITE) :
   cdef int i,j,k,j1,j2,j3,k1,k2,it
   cdef int dim1,dim2,dim3
   cdef r_2,s_2
   cdef double Z,nume,denom,sum1,val
   dim1 = data.shape[0]
   dim2 = data.shape[1]
   dim3 = data.shape[2]
   if r % 2 == 0 :
      r = r + 1
   if s % 2 == 0 :
      s = s + 1
   r_2 = int(r/2)
   s_2 = int(s/2)
   data_bar  = np.array(np.zeros((dim1,dim2,dim3)),dtype=np.double)
   U         = np.array(np.zeros((dim1,dim2,dim3,K)),dtype=np.double)
   V_next    = np.array(V,dtype=np.double)
   V         = V - np.array(np.ones(K),dtype=np.double)
   x_i       = np.array(np.zeros(int(s**2)),dtype=np.double)
   x_j       = np.array(np.zeros(int(s**2)),dtype=np.double)
   #  obtain the non-local means-filtered image
   start_time = time.time()
   for i in range(r_2+s_2,dim1-r_2-s_2):
    for j in range(r_2+s_2,dim2-r_2-s_2):
     for k in range(0,dim3): #(r_2+s_2,dim3-r_2-s_2):
      if data[i,j,k] > 0 :
         Z = 0
         for j1 in range(-r_2,r_2+1):
          for j2 in range(-r_2,r_2+1):
             #for j3 in range(-r_2,r_2+1):
             x_i = np.reshape( data[(i-s_2):(i+s_2+1),(j-s_2):(j+s_2+1),k], int(s**2) )
             x_j = np.reshape( data[(i+j1-s_2):(i+j1+s_2+1),(j+j2-s_2):(j+j2+s_2+1),k], int(s**2) )
             Z = Z + math.exp( - np.linalg.norm( x_i - x_j ) / (h**2) )
         sum1 = 0
         for j1 in range(-r_2,r_2+1):
          for j2 in range(-r_2,r_2+1):
             #for j3 in range(-r_2,r_2+1):
             x_i = np.reshape( data[(i-s_2):(i+s_2+1),(j-s_2):(j+s_2+1),k], int(s**2) )
             x_j = np.reshape( data[(i+j1-s_2):(i+j1+s_2+1),(j+j2-s_2):(j+j2+s_2+1),k], int(s**2) )
             sum1 = sum1 + (math.exp( - np.linalg.norm( x_i - x_j ) / (h**2) ) / Z) * data[i+j1,j+j2,k]
         data_bar[i,j,k] = sum1
         #print "FCM_NLS NL-means filter, index {} out of {}. Time={} seconds.".format(i*dim2*dim3+j*dim3+k,\
         #                   (dim1-1)*dim2*dim3+(dim2-1)*dim3+(dim3-1) ,time.time() - start_time)
   print "FCM_NLS NL-means filter. Time={} seconds.".format(time.time() - start_time)
   #  apply iterative algorithm
   it = 1
   while it <= MAX_ITE : #np.linalg.norm(V_next - V) >= tole and it <= MAX_ITE :
      V = V_next
      # update the membership functions
      for i in range(0,dim1):
       for j in range(0,dim2):
        for k in range(0,dim3):
          if data[i,j,k] > 0 :
             for k1 in range(0,K):
                sum1 = 0
                for k2 in range(0,K):
                   nume = (data[i,j,k] - V[k1])**2  +  beta*(data_bar[i,j,k] - V[k1])**2
                   denom = (data[i,j,k] - V[k2])**2  +  beta*(data_bar[i,j,k] - V[k2])**2
                sum1 = sum1 + nume/denom
                U[i,j,k,k1] = 1.0 / ( sum1**(1.0/(m-1)) )
      # update the cluster centers
      for k1 in range(0,K):
         nume = 0
         denom = 0
         for i in range(0,dim1):
          for j in range(0,dim2):
           for k in range(0,dim3):
              if data[i,j,k] > 0 :
                 nume = nume + (U[i,j,k,k1]**m)*(data[i,j,k] + beta*data_bar[i,j,k])
                 denom = denom + (U[i,j,k,k1]**m)
         V_next[k1] = nume / ((1.0+beta)*denom)
      print "------FCM_NLS, iteration {} done.".format(it)
      it = it + 1
   del data_bar
   # final labeling according to membership functions
   SegMap  = np.array(np.zeros((dim1,dim2,dim3)),dtype=np.int64)
   for i in range(0,dim1):
    for j in range(0,dim2):
     for k in range(0,dim3):
        if data[i,j,k] > 0 :
           k2  = 0
           val = U[i,j,k,k2]
           for k1 in range(1,K):
             if U[i,j,k,k1] > val :
                val = U[i,j,k,k1]
                k2 = k1
           SegMap[i,j,k] = labels[k2]
   
   return SegMap

