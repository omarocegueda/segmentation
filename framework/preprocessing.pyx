cimport cython
import numpy as np
cimport numpy as np
import scipy as sp
import pickle
import os
import dipy.viz.regtools as rt
from PIL import Image
#from nipype.interfaces.ants import N4BiasFieldCorrection
from dipy.align import VerbosityLevels
from dipy.align.transforms import regtransforms
from dipy.align.imaffine import (AffineMap,
                                 transform_centers_of_mass,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.metrics import CCMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.reslice import reslice
from skimage.morphology import ball,opening,closing,dilation,erosion,watershed
from scipy import ndimage
from fast_morph import (SequencialSphereDilation,
                                  create_sphere,
                                  get_list,
                                  get_subsphere_lists,
                                  isotropic_erosion,
                                  isotropic_dilation)

##----------------------------------------------------------------------------------
##----------------------------------------------------------------------------------

# Normalize intensity values to the range [0,scaleVal]
# Input:  data numpy array with the intensity values to normalize and
#         scaleVal value for the normalization.
# Output: data numpy array with the normalized intensity values.
def NormalizeIntensity(data,scaleVal):
   maxVal = data.max()
   minVal = data.min()
   data = (scaleVal/(maxVal-minVal))*(data-minVal)
   return data

##----------------------------------------------------------------------------------
##----------------------------------------------------------------------------------

def dipy_align(static, static_grid2world, moving, moving_grid2world,
               transforms=None, level_iters=None, prealign=None):
    r''' Full rigid registration with Dipy's imaffine module
    Here we implement an extra optimization heuristic: move the geometric
    centers of the images to the origin. Imaffine does not do this by default
    because we want to give the user as much control of the optimization
    process as possible.
    '''
    # Bring the center of the moving image to the origin
    c_moving = tuple(0.5 * np.array(moving.shape, dtype=np.float64))
    c_moving = moving_grid2world.dot(c_moving+(1,))
    correction_moving = np.eye(4, dtype=np.float64)
    correction_moving[:3,3] = -1 * c_moving[:3]
    centered_moving_aff = correction_moving.dot(moving_grid2world)
    # Bring the center of the static image to the origin
    c_static = tuple(0.5 * np.array(static.shape, dtype=np.float64))
    c_static = static_grid2world.dot(c_static+(1,))
    correction_static = np.eye(4, dtype=np.float64)
    correction_static[:3,3] = -1 * c_static[:3]
    centered_static_aff = correction_static.dot(static_grid2world)
    dim = len(static.shape)
    metric = MutualInformationMetric(nbins=32, sampling_proportion=0.3)
    #metric = LocalCCMetric(radius=4)
    #metric.verbosity = VerbosityLevels.DEBUG
    # Registration schedule: center-of-mass then translation, then rigid and then affine
    if prealign is None:
        prealign = 'mass'
    if transforms is None:
        transforms = ['TRANSLATION', 'RIGID', 'AFFINE']
    nlevels = len(transforms)
    if level_iters is None:
        level_iters = [[10000, 1000, 100]] * nlevels
    sol = np.eye(dim + 1)
    for i in range(nlevels):
        transform_name = transforms[i]
        affr = AffineRegistration(metric=metric, level_iters=level_iters[i])
        affr.verbosity = VerbosityLevels.DEBUG
        transform = regtransforms[(transform_name, dim)]
        print('Optimizing: %s'%(transform_name,))
        x0 = None
        sol = affr.optimize(static, moving, transform, x0,
                              centered_static_aff, centered_moving_aff, starting_affine = prealign)
        prealign = sol.affine.copy()
    # Now bring the geometric centers back to their original location
    fixed = np.linalg.inv(correction_moving).dot(sol.affine.dot(correction_static))
    sol.set_affine(fixed)
    sol.domain_grid2world = static_grid2world
    sol.codomain_grid2world = moving_grid2world
    return sol

#
# Rigid register between two MRI images.
# Input:  data_fix numpy array with the fixed 3D MRI image to which
#         it is desired to find a transformation, data_mov numpy array
#         with the moving 3D MRI image which is desired to be rigidly
#         transformed to the data_fix space, aff_fix affine transform
#         corresponding to the nifty image of the data_fix data, aff_mov
#         affine transform corresponding to the nifty image of the
#         data_mov data, zooms zooms_fix corresponding to the physical
#         voxel spacing in the nifty image of the data_fix data, zooms
#         zooms_mov corresponding to the physical voxel spacing in the
#         nifty image of the data_mov data, output_fname string with the
#         desired output file name for storing the transform of the rigid
#         register and type of scale ts desired for the initial transform
#         (ts=1 for identity, ts=2 for isotropic kind of scaling, ts=3 for
#         None).
# Output: rT rigid transform that rigidly maps from data_mov to data_fix.
def rigidRegister(np.ndarray[np.double_t, ndim=3] data_fix, \
                   np.ndarray[np.double_t, ndim=3] data_mov, \
                   np.ndarray[np.double_t, ndim=2] aff_fix, \
                   np.ndarray[np.double_t, ndim=2] aff_mov, \
                   np.ndarray[np.double_t, ndim=1] zooms_fix, \
                   np.ndarray[np.double_t, ndim=1] zooms_mov, str output_fname, int ts) :
   cdef double denom
   if ts == 1 :
      scale = np.eye(4,dtype=np.double)
   elif ts == 2 :
      scale = np.eye(4,dtype=np.double)
      denom = data_fix.shape[0] * zooms_fix[0]
      denom = denom * data_fix.shape[1] * zooms_fix[1]
      denom = denom * data_fix.shape[2] * zooms_fix[2]
      nume  = data_mov.shape[0] * zooms_mov[0]
      nume  = nume * data_mov.shape[1] * zooms_mov[1]
      nume  = nume * data_mov.shape[2] * zooms_mov[2]
      iso_scale = (np.double(nume)/np.double(denom))**(1.0/3)
      print "iso_scale = {}".format(iso_scale)
      scale[:3,:3] *= iso_scale
   else :
      scale = None
   if os.path.isfile(output_fname) :
      rT = pickle.load(open(output_fname,'r'))
   else :
      transforms  = ["RIGID","AFFINE"]
      level_iters = [[10000, 1000, 100], [100]]
      rT          = dipy_align(data_fix,aff_fix, data_mov,aff_mov,
                    transforms = transforms, level_iters = level_iters, prealign=scale)
      pickle.dump(rT,open(output_fname,'w'))
   return rT


#
# Diffeomorphic register between two MRI images.
# Input:  data_fix numpy array with the fixed 3D MRI image to which
#         it is desired to find a transformation, data_mov numpy array
#         with the moving 3D MRI image which is desired to be transformed
#         in a diffeomorphic way to the data_fix space, aff_fix affine
#         transform corresponding to the nifty image of the data_fix data,
#         aff_mov affine transform corresponding to the nifty image of the
#         data_mov data, rT starting transform (recommended based on
#         rigid transform returned by the rigidRegister function applied to
#         the same data arrays) and output_fname string with the desired
#         output file name for storing the transform of the diffeomorphic
#         register.
# Output: dT diffeomorphic transform that diffeomorphically maps from data_mov
#         to data_fix.
def diffeomorphicRegister(np.ndarray[np.double_t, ndim=3] data_fix, \
                   np.ndarray[np.double_t, ndim=3] data_mov, \
                   np.ndarray[np.double_t, ndim=2] aff_fix, \
                   np.ndarray[np.double_t, ndim=2] aff_mov, rT, str output_fname ) :
   scale = np.eye(4,dtype=np.double)
   if rT == None :
      prealignT = scale
   else :
      prealignT = rT.affine
   
   if os.path.isfile(output_fname) :
      dT = pickle.load(open(output_fname,'r'))
   else :
      metric = CCMetric(3)
      sdr    = SymmetricDiffeomorphicRegistration(metric)
      dT     = sdr.optimize(data_fix, data_mov, \
                  aff_fix, aff_mov, prealign=prealignT)
      pickle.dump(dT,open(output_fname,'w'))
   return dT

##----------------------------------------------------------------------------------
##----------------------------------------------------------------------------------

#
# Intracranial Cavity Extraction (ICE).
# Input:  data numpy array with the 3D MRI image whose ICE is desired,
#         numpy array with the atlas ICE mask atlasMask (to aid in the
#         process), output directory name output_dir for printing
#         temporal slices of the process, base slice nSlice to show
#         during the process and dimensions of the data array dim1, dim2,
#         dim3.
# Output: ICEMask numpy array with the ICE mask, gradOp numpy
#         array with the morphological gradient of the opening of
#         the data array to aid in later stages of the pipeline and
#         C centroid of the atlasMask data array (also to aid in
#         later stages).
def ICE(np.ndarray[np.double_t, ndim=3] data, \
         np.ndarray[np.int_t, ndim=3] atlasMask, str output_dir, \
         int nSlice, int dim1, int dim2, int dim3) :
   cdef int i,j,k,x,y,z,rs
   cdef double val
   #   apply atlas head mask
   mask1 = np.array(np.zeros((dim1,dim2,dim3)),dtype=np.int32)
   mask1[ atlasMask > 0 ] = 1
   #   define structuring element
   struct_elem = ball(9)
   #   Perform opening of morphologic erosion of mask
   #maskOpenedEroded  = opening(erosion(mask1,struct_elem),ball(4))
   #Im = Image.fromarray(np.uint8(maskOpenedEroded[:,:,nSlice]*255.0))
   #Im.save(output_dir+'PREP_maskEroded_and_Opened.png')
   
   #   Perform morphologic dilation of mask
   mask1 = dilation(mask1,struct_elem)
   data[ mask1 == 0 ] = 0
   
   Im = Image.fromarray(np.uint8(mask1[:,:,nSlice]*255.0))
   Im.save(output_dir+'PREP_mask1.png')
   
   #   obtain bounding box coordinates of brain mask to serve as
   #   landmarks for unbiased Watershed markers relative to center of mass marker
   #minI,maxI,minJ,maxJ,minK,maxK = BoundingBoxCoords(mask1, 1, dim1,dim2,dim3)
   
   #   define structuring element
   struct_elem = ball(4) # <-- should have 9 voxel units of diameter
   #   Perform morphologic opening on T2 image
   opened = opening(data,struct_elem)
   Im = Image.fromarray(np.uint8(opened[:,:,nSlice]*255.0))
   Im.save(output_dir+'PREP_openedT2.png')
   
   #   Obtain morphological gradient of opened data image
   cross_se = ball(1) # <-- should have 3 voxel units of diameter
   dilationO = dilation(opened,cross_se)
   erosionO  = erosion(opened,cross_se)
   del opened
   gradOp = dilationO - erosionO
   Im = Image.fromarray(np.uint8(gradOp[:,:,nSlice]*255.0))
   Im.save(output_dir+'PREP_GradientMorp_NoNorm.png')
   del dilationO
   del erosionO
   gradOp = NormalizeIntensity(gradOp,255.0)
   Im = Image.fromarray(np.uint8(gradOp[:,:,nSlice]))
   Im.save(output_dir+'PREP_GradientMorp_Norm.png')
   
   #   Obtain segmentation function (sum of increasing scale dilations)
   SSD             = SequencialSphereDilation(gradOp)
   nScaleDilations = 5 # counts dilation at 0 radius
   dilGradOp       = gradOp
   for r in range(1,nScaleDilations):
      SSD.expand(gradOp)
      dilGradOp = SSD.get_current_dilation() + dilGradOp
   
   del SSD
   segFuncGOp  = NormalizeIntensity(dilGradOp,255.0)
   Im = Image.fromarray(np.uint8(segFuncGOp[:,:,nSlice]))
   Im.save(output_dir+'PREP_seg_func_ICE.png')
   del dilGradOp
   
   #   Obtain gravity center of mask of T2
   C    = np.zeros(3,dtype=np.float)
   #CenM = ndimage.measurements.center_of_mass(atlasMask)
   #C[0] = CenM[0]
   #C[1] = CenM[1]
   #C[2] = CenM[2]
   #print "Centroid = {}".format(C)
   
   #   set two class of markers (for marker based watershed segmentation)
   markersICE = np.array(np.zeros((dim1,dim2,dim3)),dtype=np.int32)
   #markersICE[ maskOpenedEroded==1 ] = 2
   markersICE[ atlasMask==2 ] = 2
   markersICE = opening(markersICE,struct_elem)
   markersICE[ mask1==0 ] = 1
   #rs = 11
   #val = min( min( min(abs(maxI-C[0]),abs(minI-C[0])) , min(abs(maxJ-C[1]),abs(minJ-C[1])) ),  min(abs(maxK-C[2]),abs(minK-C[2])) )
   #rs = int(0.75 * val)
   #if rs % 2 == 0 :
   #   rs = rs + 1
   #print "Sphere marker diameter = {}".format(rs)
   #markersphere = ball(int(rs/2))
   #for i in range(0,rs):
   # for j in range(0,rs):
   #  for k in range(0,rs):
   #     markersICE[int(C[0]-(rs/2))+i,int(C[1]-(rs/2))+j,int(C[2]-(rs/2))+k] = markersphere[i,j,k]
   
   #for y in range(minJ,maxJ+1):
   #  for z in range(minK,maxK+1):
   #     markersICE[minI,y,z] = 1
   #     markersICE[maxI,y,z] = 1
   #
   #for x in range(minI,maxI+1):
   #  for z in range(minK,maxK+1):
   #     markersICE[x,minJ,z] = 1
   #     markersICE[x,maxJ,z] = 1
   #
   #for y in range(minJ,maxJ+1):
   #  for x in range(minI,maxI+1):
   #     markersICE[x,y,minK] = 1
   #     markersICE[x,y,maxK] = 1
   
   #for y in range(0,dim2):
   #  for z in range(0,dim3):
   #     if mask1[0,y,z] == 0 :
   #        markersICE[0,y,z] = 1
   #     if mask1[dim1-1,y,z] == 0 :
   #        markersICE[dim1-1,y,z] = 1
   #
   #for x in range(0,dim1):
   #  for z in range(0,dim3):
   #     if mask1[x,0,z] == 0 :
   #        markersICE[x,0,z] = 1
   #     if mask1[x,dim2-1,z] == 0 :
   #        markersICE[x,dim2-1,z] = 1
   #
   #for y in range(0,dim2):
   #  for x in range(0,dim1):
   #     if mask1[x,y,0] == 0 :
   #        markersICE[x,y,0] = 1
   #     if mask1[x,y,dim3-1] == 0 :
   #        markersICE[x,y,dim3-1] = 1
   
   #   Apply watershed segmentation with markers
   segFuncGOp = np.array(segFuncGOp,dtype=np.int32)
   ICEMask = watershed(segFuncGOp,markersICE)
   del segFuncGOp
   del markersICE
   ICEMask = dilation(ICEMask,ball(1))
   
   return ICEMask, gradOp, C


#
# Bounding box coordinates.
# Input:  data int numpy array with the 3D image whose bounding box
#         is desired, value of data array val which forms the object of
#         interest for the bounding box and dimensions of the data array
#         dim1, dim2, dim3.
# Output: bounding box coordinates of the data array denoted with minimum
#         or maximum for each axis of the data array (i,j,k), that is
#         minI,maxI,minJ,maxJ,minK,maxK.
@cython.boundscheck(False)
def BoundingBoxCoords(np.ndarray[int, ndim=3] data, int val, \
         int dim1, int dim2, int dim3 ) :
   cdef int maxI = 0
   cdef int maxJ = 0
   cdef int maxK = 0
   cdef int minI = dim1
   cdef int minJ = dim2
   cdef int minK = dim3
   cdef int i
   cdef int j
   cdef int k
   for i in range(0,dim1):
     for j in range(0,dim2):
      for k in range(0,dim3):
         if data[i,j,k] == val :
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
   
   return minI,maxI,minJ,maxJ,minK,maxK


