import numpy as np
import scipy as sp
import nibabel as nib
import os
import argparse
import time
import dipy.viz.regtools as rt
import matplotlib.pyplot as plt
import preprocessing
import segmentation
import evaluation
from PIL import Image
from pylab import *
from skimage.morphology import ball,opening,closing,dilation,erosion,watershed
from scipy.ndimage.measurements import watershed_ift
from skimage.restoration import denoise_bilateral
from dipy.align.reslice import reslice
from scipy import ndimage
from fast_morph import (SequencialSphereDilation,
                                  create_sphere,
                                  get_list,
                                  get_subsphere_lists,
                                  isotropic_erosion,
                                  isotropic_dilation)



# input parameters to the program
parser = argparse.ArgumentParser(description='Segmentation tool for NIFTY MRI infant brain images.')
parser.add_argument('inFile', metavar='inFile', type=str, \
                 help='Name of the text file where the directories and labels are.')
parser.add_argument('outFile', metavar='outFile', type=str, \
                 help='Name for the output NIFTY file with the segmentation.')
parser.add_argument('output_textfile1', metavar='output_textfile1', type=str, \
                 help='Name for the output textfile with the DICE coefficients results.')
parser.add_argument('output_textfile2', metavar='output_textfile2', type=str, \
                 help='Name for the output textfile with the Accuracy results.')
parser.add_argument('scheme', metavar='scheme', type=int, \
                 help='Algorithm scheme for benchmarking. 1 - K-Means, 2 - FCM, 3 - PFCM, 4 - FCM_NLS, 5 - Proposed, 6 - Proposed + PFCM, 7 - Proposed + FCM_NLS.')
parser.add_argument('type_alineation', metavar='type_alineation', type=int, \
                 help='Type of alineation to carry out the registration process. 1 - T2-to-T2, 2 - T1-to-T2, 3 - T2-to-T1, 4 - T1-to-T1')
parser.add_argument('tsRigid', metavar='tsRigid', type=int, \
                 help='Type of scaling for the initial transform of the Rigid Register (in Diffeomorphic phase). 1 - Identity, 2 - Isotropic, 3 - None')
parser.add_argument('par_gamma', metavar='par_gamma', type=np.double, \
                 help='Proportion of Hessian filter parameters in each exponential tuning curve in range(0,1]')
parser.add_argument('par_priorthresh', metavar='par_priorthresh', type=np.double, \
                 help='Hessian threshold parameter for CGM detection in range (0,1].')
parser.add_argument('par_w', metavar='par_w', type=np.double, \
                 help='CGM enhancing weight value in range (0,1].')

parser.add_argument('par_nln', metavar='par_nln', type=int, \
                 help='Number of local neighbours for SGM enhancing.')
parser.add_argument('par_nnln', metavar='par_nnln', type=int, \
                 help='Number of non-local neighbours for SGM enhancing.')
parser.add_argument('par_r', metavar='par_r', type=int, \
                 help='Non-local window size for SGM enhancing.')
parser.add_argument('par_ps', metavar='par_ps', type=int, \
                 help='Patch size for non-local neighbours for SGM enhancing.')
parser.add_argument('par_ra', metavar='par_ra', type=int, \
                 help='If non-local neighbours are to be sampled to only 50 percent and randomly (1) or not (0).')
parser.add_argument('par_uset1', metavar='par_uset1', type=int, \
                 help='Whether to use T1 image for GM enhancing (1) or not (0).')

parser.add_argument('apply_sgmenhancing', metavar='apply_sgmenhancing', type=int, \
                 help='Whether to apply SGM enhancing (1) or not (0).')
parser.add_argument('apply_cgmenhancing', metavar='apply_cgmenhancing', type=int, \
                 help='Whether to apply CGM enhancing (1) or not (0).')


args = parser.parse_args()
directories_and_labels_file = args.inFile
output_SegFile              = args.outFile
output_textfile1            = args.output_textfile1
output_textfile2            = args.output_textfile2
algorithm_scheme_id         = args.scheme
type_alineation             = args.type_alineation
tsRigid                     = args.tsRigid

par_gamma                   = args.par_gamma
par_priorthresh             = args.par_priorthresh
par_w                       = args.par_w

par_nln                     = args.par_nln
par_nnln                    = args.par_nnln
par_r                       = args.par_r
par_ps                      = args.par_ps
par_ra                      = args.par_ra
par_uset1                   = args.par_uset1

apply_sgmenhancing          = args.apply_sgmenhancing
apply_cgmenhancing          = args.apply_cgmenhancing

# read directories and atlas label
base_dir    = ' '
neo_subject = ' '
results_dir = ' '
middir1     = ' '
middir2     = ' '
with open(directories_and_labels_file) as fp :
   i = 0
   for line in fp :
      if i == 0 :
         base_dir = line[0:(len(line)-1)]
      elif i == 1 :
         neo_subject = line[0:(len(line)-1)]
      elif i == 2 :
         results_dir = line[0:(len(line)-1)]
      elif i == 3 :
            middir1 = line[0:(len(line)-1)]
      else :
         if i == 4 :
            middir2 = line[0:(len(line)-1)]
      i = i + 1



# Read atlas and subject files

t2CurrentSubjectName = base_dir + middir1 + neo_subject + 'T2_1-1.nii.gz'
t1CurrentSubjectName = base_dir + middir1 + neo_subject + 'T1_1-1.nii.gz'

t2CurrentSubject = nib.load(t2CurrentSubjectName)
t1CurrentSubject = nib.load(t1CurrentSubjectName)

t2CurrentSubject_data = t2CurrentSubject.get_data()
t1CurrentSubject_data = t1CurrentSubject.get_data()

t2CSAffine = t2CurrentSubject.get_affine()
t1CSAffine = t1CurrentSubject.get_affine()

#GTName                = base_dir + middir1 +neo_subject+'manualSegm.nii.gz'
GTName                = base_dir + middir1 +neo_subject+'T2.nii.gz'
GT_data               = nib.load(GTName).get_data()



file1Name   = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+"T2_1-1_ICE_isovox.nii.gz"
file2Name   = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+"T1_1-2_ICE_isovox.nii.gz"
file3Name   = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+"training_T2_ICE_isovox.nii.gz"
file4Name   = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+"training_T1_ICE_isovox.nii.gz"
file5Name   = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+"training_Segm_ICE_isovox.nii.gz"
file1Exists = os.path.isfile(file1Name)
file2Exists = os.path.isfile(file2Name)
file3Exists = os.path.isfile(file3Name)
file4Exists = os.path.isfile(file4Name)
file5Exists = os.path.isfile(file5Name)

if file1Exists and file2Exists and file3Exists and file4Exists and file5Exists :
   t2CurrentSubject_data = nib.load(file1Name).get_data()
   t1CurrentSubject_data = nib.load(file2Name).get_data()
   TT2_data              = nib.load(file3Name).get_data()
   TT1_data              = nib.load(file4Name).get_data()
   TSegm_data            = nib.load(file5Name).get_data()
   
   zoomsT2CS  = nib.load(file1Name).get_header().get_zooms()[:3]
   
   dim1 = t2CurrentSubject_data.shape[0]
   dim2 = t2CurrentSubject_data.shape[1]
   dim3 = t2CurrentSubject_data.shape[2]
   nSliceGT = int(GT_data.shape[2] / 2)
   difSliceGT = int(GT_data.shape[2]/10)
   nSlice = int(round(float(dim3*nSliceGT)/float(GT_data.shape[2])))
   difSlice = int(round(float(dim3*difSliceGT)/float(GT_data.shape[2])))
   minI = 0
   maxI = dim1
   minJ = 0
   maxJ = dim2
   minK = 0
   maxK = dim3
   with open(t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+"BBCoords.txt") as fp :
      i = 0
      for line in fp :
         if i == 0 :
            minI = int(line[0:(len(line)-1)])
         elif i == 1 :
            maxI = int(line[0:(len(line)-1)])
         elif i == 2 :
            minJ = int(line[0:(len(line)-1)])
         elif i == 3 :
            maxJ = int(line[0:(len(line)-1)])
         elif i == 4 :
            minK = int(line[0:(len(line)-1)])
         else :
            if i == 5 :
               maxK = int(line[0:(len(line)-1)])
         i = i + 1
   start_time = time.time()
else :
   # Step 1.1 - Intensity inhomogeneity correction (currently done with 3D Slicer Version 3.6.3)


   #    1.2.1 - Rigid Registration of T1 subject image to T2 subject image
   #    ---------------------------------------------------------------------------------------------


   start_time = time.time()

   rigidTransformName = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))] +"/T1_towards_T2_rigid.p"

   rigidTransform = preprocessing.rigidRegister(np.asarray(t2CurrentSubject_data,dtype=np.double), np.asarray(t1CurrentSubject_data,dtype=np.double), \
                      t2CSAffine, t1CSAffine, None, None, rigidTransformName, 1)

   print "Rigid register: {} seconds.".format(time.time() - start_time)

   t1CurrentSubject_data1 = rigidTransform.transform(t1CurrentSubject_data)
   rt.overlay_slices(t2CurrentSubject_data, t1CurrentSubject_data1, slice_type=2, slice_index=25, ltitle='T2 Subject', rtitle='T1 Subject', fname= results_dir + 'PREP_T1_to_T2_rigid_coronal_slice25.png')
   T1CS = nib.Nifti1Image(t1CurrentSubject_data1,t2CSAffine)
   nib.save(T1CS,t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/T1_1-2.nii.gz")

   del t1CurrentSubject_data1
   del T1CS
   del t1CSAffine

   #    ---------------------------------------------------------------------------------------------


   #    1.2.2 - Rigid Registration # 1 of training subjects to T2
   #    ---------------------------------------------------------------------------------------------

   t2CurrentSubjectName     = base_dir + middir1 +neo_subject+ 'T2_1-1.nii.gz'
   t1CurrentSubjectName     = base_dir + middir1 +neo_subject+ 'T1_1-2.nii.gz'
   t2TrainingSubjectName    = base_dir + middir1 + neo_subject + 'training_T2.nii.gz'
   t1TrainingSubjectName    = base_dir + middir1 + neo_subject + 'training_T1.nii.gz'
   trainingSegmName         = base_dir + middir1 + neo_subject + 'training_SEGM.nii.gz'


   t2CurrentSubject     = nib.load(t2CurrentSubjectName)
   t1CurrentSubject     = nib.load(t1CurrentSubjectName)
   t2TrainingSubject    = nib.load(t2TrainingSubjectName)
   t1TrainingSubject    = nib.load(t1TrainingSubjectName)
   trainingSegm         = nib.load(trainingSegmName)


   t2CurrentSubject_data     = t2CurrentSubject.get_data()
   t1CurrentSubject_data     = t1CurrentSubject.get_data()
   t2TrainingSubject_data    = t2TrainingSubject.get_data()
   t1TrainingSubject_data    = t1TrainingSubject.get_data()
   trainingSegm_data         = trainingSegm.get_data()


   t2CSAffine    = t2CurrentSubject.get_affine()
   t2TrSAffine   = t2TrainingSubject.get_affine()

   zoomsT2CS  = t2CurrentSubject.get_header().get_zooms()[:3]
   zoomsT2TrS = t2TrainingSubject.get_header().get_zooms()[:3]


   start_time = time.time()

   rigidTransformName = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))] + "/training_towards_neo_rigid1.p"

   t2TrainingSubject_data2 = t2TrainingSubject_data
   t2TrainingSubject_data2[ trainingSegm_data == 0 ] = 0
   rigidTransform = preprocessing.rigidRegister(np.asarray(t2CurrentSubject_data,dtype=np.double), np.asarray(t2TrainingSubject_data2,dtype=np.double), \
                      t2CSAffine, t2TrSAffine, np.asarray(zoomsT2CS,dtype=np.double), \
                      np.asarray(zoomsT2TrS,dtype=np.double), rigidTransformName, tsRigid)


   print "Rigid register #1 (training subject to subject): {} seconds.".format(time.time() - start_time)

   t2TrainingSubject_data2    = rigidTransform.transform(t2TrainingSubject_data2)
   trainingSegm_data2  = rigidTransform.transform(np.asarray(trainingSegm_data,dtype=np.float),'nearest')
   rt.overlay_slices(t2CurrentSubject_data, t2TrainingSubject_data2, slice_type=2, slice_index=25, ltitle='T2 Subject', rtitle='T2 Training', fname= results_dir + 'PREP_Training_T2_to_neo_rigid1_reg_coronal_slice25.png')


   del t2TrainingSubject_data2

   #    ---------------------------------------------------------------------------------------------

   start_time = time.time()


   #    1.3 - Resampling for isotropic voxels
   #    ---------------------------------------------------------------------------------------------

   n_zooms = (zoomsT2CS[0],zoomsT2CS[0],zoomsT2CS[0])
   t2CurrentSubject_data,t2CSAffine   = reslice(t2CurrentSubject_data,t2CSAffine,zoomsT2CS,n_zooms)
   #t2TrainingSubject_data,t2TrSAffine = reslice(t2TrainingSubject_data,t2TrSAffine,zoomsT2TrS,n_zooms)
   t1CurrentSubject_data,_            = reslice(t1CurrentSubject_data,t2CSAffine,zoomsT2CS,n_zooms)
   trainingSegm_data2,_               = reslice(trainingSegm_data2,t2TrSAffine,zoomsT2TrS,n_zooms,order=0)


   #    1.4 - Anisotropic diffusion filter
   #    ---------------------------------------------------------------------------------------------

   scaleValue = 1.0

   t2CurrentSubject_data  = denoise_bilateral(preprocessing.NormalizeIntensity(t2CurrentSubject_data,scaleValue),win_size=5)
   t2TrainingSubject_data = denoise_bilateral(preprocessing.NormalizeIntensity(t2TrainingSubject_data,scaleValue),win_size=5)
   t1CurrentSubject_data  = denoise_bilateral(preprocessing.NormalizeIntensity(t1CurrentSubject_data,scaleValue),win_size=5)
   t1TrainingSubject_data = denoise_bilateral(preprocessing.NormalizeIntensity(t1TrainingSubject_data,scaleValue),win_size=5)

   ##    Normalize the rest of the volume intensity values to [0,255]
   #scaleValue            = 255.0
   #trainingSegm_data2    = preprocessing.NormalizeIntensity(trainingSegm_data2,scaleValue)

   dim1 = t2CurrentSubject_data.shape[0]
   dim2 = t2CurrentSubject_data.shape[1]
   dim3 = t2CurrentSubject_data.shape[2]

   nSliceGT = int(GT_data.shape[2] / 2)
   difSliceGT = int(GT_data.shape[2]/10)
   nSlice = int(round(float(dim3*nSliceGT)/float(GT_data.shape[2])))
   difSlice = int(round(float(dim3*difSliceGT)/float(GT_data.shape[2])))

   #    1.5 - Intracranial Cavity Extraction
   #    ---------------------------------------------------------------------------------------------

   ICEMask, gradientOT2, C = preprocessing.ICE(np.asarray(t2CurrentSubject_data,dtype=np.double), \
                                                np.asarray(trainingSegm_data2,dtype=np.int_), \
                                                results_dir,nSlice,dim1,dim2,dim3)


   #   Apply Inctracranial Cavity Extraction with segmented watershed mask
   t2CurrentSubject_data[ ICEMask == 1 ] = 0
   t1CurrentSubject_data[ ICEMask == 1 ] = 0

   t2TrainingSubject_data[ trainingSegm_data == 0 ] = 0
   t1TrainingSubject_data[ trainingSegm_data == 0 ] = 0


   #   show a sample resulting slice

   Im = Image.fromarray(np.uint8(ICEMask[:,:,nSlice]*127))
   Im.save(results_dir+'PREP_ICEMask.png')
   Im = Image.fromarray(np.uint8(ICEMask[:,:,nSlice+difSlice]*127))
   Im.save(results_dir+'PREP_ICEMask_p'+str(difSlice)+'.png')
   Im = Image.fromarray(np.uint8(ICEMask[:,:,nSlice-difSlice]*127))
   Im.save(results_dir+'PREP_ICEMask_m'+str(difSlice)+'.png')
   Im = Image.fromarray(np.uint8(t1CurrentSubject_data[:,:,nSlice]*255.0))
   Im.save(results_dir+'PREP_t1CS.png')
   Im = Image.fromarray(np.uint8(t1CurrentSubject_data[:,:,nSlice+difSlice]*255.0))
   Im.save(results_dir+'PREP_t1CS_p'+str(difSlice)+'.png')
   Im = Image.fromarray(np.uint8(t1CurrentSubject_data[:,:,nSlice-difSlice]*255.0))
   Im.save(results_dir+'PREP_t1CS_m'+str(difSlice)+'.png')
   Im = Image.fromarray(np.uint8(t2CurrentSubject_data[:,:,nSlice]*255.0))
   Im.save(results_dir+'PREP_t2CS.png')
   Im = Image.fromarray(np.uint8(t2CurrentSubject_data[:,:,nSlice+difSlice]*255.0))
   Im.save(results_dir+'PREP_t2CS_p'+str(difSlice)+'.png')
   Im = Image.fromarray(np.uint8(t2CurrentSubject_data[:,:,nSlice-difSlice]*255.0))
   Im.save(results_dir+'PREP_t2CS_m'+str(difSlice)+'.png')


   #   Get bounding box coordinates to reduce computations

   minI,maxI,minJ,maxJ,minK,maxK = preprocessing.BoundingBoxCoords(ICEMask,2,dim1,dim2,dim3)

   print "bounding box i:(min={},max={}), j:(min={},max={}), k:(min={},max={}).".format(minI,maxI,minJ,maxJ,minK,maxK)

   print "Until ICE: {} seconds.".format(time.time() - start_time)

   del ICEMask

   if type_alineation == 1 : # T2-to-T2
      rigidTrainingTransformName = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))] + "/trainingT2_towards_neoT2_rigid_afterICE.p"
      rigidTrainingTransform = preprocessing.rigidRegister(np.asarray(t2CurrentSubject_data,dtype=np.double), \
                         np.asarray(t2TrainingSubject_data,dtype=np.double), \
                         t2CSAffine, t2TrSAffine, np.asarray(zoomsT2CS,dtype=np.double), \
                         np.asarray(zoomsT2TrS,dtype=np.double), rigidTrainingTransformName, tsRigid)
      print "Until Rigid register after ICE (training subject to subject): {} seconds.".format(time.time() - start_time)
      TT2_data = rigidTrainingTransform.transform(t2TrainingSubject_data)
      rt.overlay_slices(t2CurrentSubject_data, TT2_data, slice_type=2, slice_index=nSlice, ltitle='T2 Subject', rtitle='T2 Training', fname= results_dir + 'PREP_Training_T2_to_neoT2_rigid_reg_afterICE_coronal_slice'+str(nSlice)+'.png')
      diff_training_map_name = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))] + "/trainingT2_towards_neoT2_diff_afterICE.p"
      diff_training_map = preprocessing.diffeomorphicRegister(np.asarray(t2CurrentSubject_data,dtype=np.double), \
                         np.asarray(t2TrainingSubject_data,dtype=np.double), \
                         t2CSAffine, t2TrSAffine, rigidTrainingTransform, diff_training_map_name )
   elif type_alineation == 2 : # T1-to-T2
      rigidTrainingTransformName = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))] + "/trainingT1_towards_neoT2_rigid_afterICE.p"
      rigidTrainingTransform = preprocessing.rigidRegister(np.asarray(t2CurrentSubject_data,dtype=np.double), \
                         np.asarray(t1TrainingSubject_data,dtype=np.double), \
                         t2CSAffine, t2TrSAffine, np.asarray(zoomsT2CS,dtype=np.double), \
                         np.asarray(zoomsT2TrS,dtype=np.double), rigidTrainingTransformName, tsRigid)
      print "Until Rigid register after ICE (training subject to subject): {} seconds.".format(time.time() - start_time)
      TT1_data = rigidTrainingTransform.transform(t1TrainingSubject_data)
      rt.overlay_slices(t2CurrentSubject_data, TT1_data, slice_type=2, slice_index=nSlice, ltitle='T2 Subject', rtitle='T1 Training', fname= results_dir + 'PREP_Training_T1_to_neoT2_rigid_reg_afterICE_coronal_slice'+str(nSlice)+'.png')
      diff_training_map_name = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))] + "/trainingT1_towards_neoT2_diff_afterICE.p"
      diff_training_map = preprocessing.diffeomorphicRegister(np.asarray(t2CurrentSubject_data,dtype=np.double), \
                         np.asarray(t1TrainingSubject_data,dtype=np.double), \
                         t2CSAffine, t2TrSAffine, rigidTrainingTransform, diff_training_map_name )
   elif type_alineation == 3 : # T2-to-T1
      rigidTrainingTransformName = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))] + "/trainingT2_towards_neoT1_rigid_afterICE.p"
      rigidTrainingTransform = preprocessing.rigidRegister(np.asarray(t1CurrentSubject_data,dtype=np.double), \
                         np.asarray(t2TrainingSubject_data,dtype=np.double), \
                         t2CSAffine, t2TrSAffine, np.asarray(zoomsT2CS,dtype=np.double), \
                         np.asarray(zoomsT2TrS,dtype=np.double), rigidTrainingTransformName, tsRigid)
      print "Until Rigid register after ICE (training subject to subject): {} seconds.".format(time.time() - start_time)
      TT2_data = rigidTrainingTransform.transform(t2TrainingSubject_data)
      rt.overlay_slices(t1CurrentSubject_data, TT2_data, slice_type=2, slice_index=nSlice, ltitle='T1 Subject', rtitle='T2 Training', fname= results_dir + 'PREP_Training_T2_to_neoT1_rigid_reg_afterICE_coronal_slice'+str(nSlice)+'.png')
      diff_training_map_name = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))] + "/trainingT2_towards_neoT1_diff_afterICE.p"
      diff_training_map = preprocessing.diffeomorphicRegister(np.asarray(t1CurrentSubject_data,dtype=np.double), \
                         np.asarray(t2TrainingSubject_data,dtype=np.double), \
                         t2CSAffine, t2TrSAffine, rigidTrainingTransform, diff_training_map_name )
   else :                      # T1-to-T1
      rigidTrainingTransformName = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))] + "/trainingT1_towards_neoT1_rigid_afterICE.p"
      rigidTrainingTransform = preprocessing.rigidRegister(np.asarray(t1CurrentSubject_data,dtype=np.double), \
                         np.asarray(t1TrainingSubject_data,dtype=np.double), \
                         t2CSAffine, t2TrSAffine, np.asarray(zoomsT2CS,dtype=np.double), \
                         np.asarray(zoomsT2TrS,dtype=np.double), rigidTrainingTransformName, tsRigid)
      print "Until Rigid register after ICE (training subject to subject): {} seconds.".format(time.time() - start_time)
      TT1_data = rigidTrainingTransform.transform(t1TrainingSubject_data)
      rt.overlay_slices(t1CurrentSubject_data, TT1_data, slice_type=2, slice_index=nSlice, ltitle='T1 Subject', rtitle='T1 Training', fname= results_dir + 'PREP_Training_T1_to_neoT1_rigid_reg_afterICE_coronal_slice'+str(nSlice)+'.png')
      diff_training_map_name = t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))] + "/trainingT1_towards_neoT1_diff_afterICE.p"
      diff_training_map = preprocessing.diffeomorphicRegister(np.asarray(t1CurrentSubject_data,dtype=np.double), \
                         np.asarray(t1TrainingSubject_data,dtype=np.double), \
                         t2CSAffine, t2TrSAffine, rigidTrainingTransform, diff_training_map_name )

   print "Until Diffeomorphic register after ICE (training subject to subject): {} seconds.".format(time.time() - start_time)

   TT2_data    = diff_training_map.transform(t2TrainingSubject_data)
   TT1_data    = diff_training_map.transform(t1TrainingSubject_data)
   TSegm_data  = diff_training_map.transform(np.asarray(trainingSegm_data,dtype=np.float),'nearest')

   # further refine ICE with diffeomorphic registration result
   t2CurrentSubject_data[ TT2_data == 0 ] = 0
   t1CurrentSubject_data[ TT2_data == 0 ] = 0

   rt.overlay_slices(t2CurrentSubject_data, TT2_data, slice_type=2, slice_index=nSlice, ltitle='T2 Subject', rtitle='T2 Training', fname= results_dir + 'PREP_Training_T2_to_neo_diff_reg_afterICE_coronal_slice'+str(nSlice)+'.png')
   rt.overlay_slices(t2CurrentSubject_data, TT1_data, slice_type=2, slice_index=nSlice, ltitle='T2 Subject', rtitle='T1 Training', fname= results_dir + 'PREP_Training_T1_to_neo_diff_reg_afterICE_coronal_slice'+str(nSlice)+'.png')
   rt.overlay_slices(t2CurrentSubject_data, TSegm_data, slice_type=2, slice_index=nSlice, ltitle='T2 Subject', rtitle='Training Segmentation', fname= results_dir + 'PREP_TrainingSegm_to_neo_diff_reg_afterICE_coronal_slice'+str(nSlice)+'.png')


   #   save partial results

   nib.save( nib.Nifti1Image(t2CurrentSubject_data,t2CSAffine), \
             t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+"T2_1-1_ICE_isovox.nii.gz")
   nib.save( nib.Nifti1Image(t1CurrentSubject_data,t2CSAffine), \
             t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+"T1_1-2_ICE_isovox.nii.gz")
   nib.save( nib.Nifti1Image(TT2_data,t2CSAffine), \
             t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+"training_T2_ICE_isovox.nii.gz")
   nib.save( nib.Nifti1Image(TT1_data,t2CSAffine), \
             t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+"training_T1_ICE_isovox.nii.gz")
   nib.save( nib.Nifti1Image(TSegm_data,t2CSAffine), \
             t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+"training_Segm_ICE_isovox.nii.gz")

   with open(t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+"BBCoords.txt", 'w') as bbox_file:
      bbox_file.write(str(minI)+'\n')
      bbox_file.write(str(maxI)+'\n')
      bbox_file.write(str(minJ)+'\n')
      bbox_file.write(str(maxJ)+'\n')
      bbox_file.write(str(minK)+'\n')
      bbox_file.write(str(maxK)+'\n')
      bbox_file.write(' ')


# 2 - SEGMENTATION----------------------------------------------------
#final_time = 0


#if    algorithm_scheme_id  ==  1  : # K-means
#   print "K-means"
#   print "--------------------------------------------------------------------"
#   print "--------------------------------------------------------------------"
#   
#elif  algorithm_scheme_id  ==  2  : # FCM
#   print "FCM"
#   print "--------------------------------------------------------------------"
#   print "--------------------------------------------------------------------"
#      
#elif  algorithm_scheme_id  ==  5  : # Proposed



Im = Image.fromarray(np.uint8(GT_data[:,:,nSliceGT])) #*31))
Im.save(results_dir+'SEGM_SegMap_GT.png')
Im = Image.fromarray(np.uint8(GT_data[:,:,nSliceGT+difSliceGT])) #*31))
Im.save(results_dir+'SEGM_SegMap_GT_p'+str(difSliceGT)+'.png')
Im = Image.fromarray(np.uint8(GT_data[:,:,nSliceGT-difSliceGT])) #*31))
Im.save(results_dir+'SEGM_SegMap_GT_m'+str(difSliceGT)+'.png')

print "Starting segmentation stage..."

if apply_cgmenhancing == 1 :
   SegMap = segmentation.CGMEnhancing(np.asarray(t2CurrentSubject_data,dtype=np.double), \
   	           np.asarray(t1CurrentSubject_data,dtype=np.double), \
                 np.asarray(TSegm_data,dtype=np.int64), \
                 results_dir, \
                 minI,minJ,minK,maxI,maxJ,maxK, nSlice,difSlice, dim1,dim2,dim3, \
                 np.double(zoomsT2CS[0]), par_gamma, par_priorthresh*255.0, par_w, 0) #par_uset1)
   final_time = time.time() - start_time
   print "Until segmentation with CGM enhancing. Time= {} seconds.".format(final_time)
else :
   # only propagate registered atlas labels
   SegMap    = np.array(np.zeros((dim1,dim2,dim3)),dtype=np.int64)
   for i in range(0,dim1):
    for j in range(0,dim2):
     for k in range(0,dim3):
       if TSegm_data[i,j,k] == 1 :
          SegMap[i,j,k] = 85
       elif TSegm_data[i,j,k] == 2 :
          SegMap[i,j,k] = 50
       elif (TSegm_data[i,j,k] == 7 or TSegm_data[i,j,k] == 8) :
          SegMap[i,j,k] = 255
       else :
          if TSegm_data[i,j,k] == 3 :
             SegMap[i,j,k] = 190
   final_time = time.time() - start_time
   print "Until segmentation with only propagation of atlas labels. Time= {} seconds.".format(final_time)

if apply_sgmenhancing == 1 :
   SegMap = segmentation.SGMSegmEnhancing(np.asarray(t2CurrentSubject_data,dtype=np.double), \
            np.asarray(t1CurrentSubject_data,dtype=np.double), \
            np.asarray(SegMap,dtype=np.int64), np.asarray(TSegm_data,dtype=np.int64), \
            np.asarray(TT2_data,dtype=np.double), np.asarray(TT1_data,dtype=np.double), \
            results_dir, \
            minI,minJ,minK,maxI,maxJ,maxK, dim3, GT_data.shape[2], \
            nSlice,difSlice, \
            par_nln, par_nnln, par_r, par_ps, par_ra, par_uset1)
   final_time = time.time() - start_time
   print "Until segmentation with SGM enhancing. Time= {} seconds.".format(final_time)
   print "--------------------------------------------------------------------"



#   Prepare segmented images for evaluation in original subject T2 space
SegMapT2Space = np.array(np.zeros((GT_data.shape[0],GT_data.shape[1],GT_data.shape[2])),dtype=np.int64)
#GT_data[ (GT_data == 4) | (GT_data == 5) | (GT_data == 6) ] = 10
#GT_data[ GT_data == 1 ] = 85
#GT_data[ GT_data == 2 ] = 50
#GT_data[ GT_data == 3 ] = 190
#GT_data[ (GT_data == 7) | (GT_data==8) ] = 255
for i in range(0,GT_data.shape[0]):
 for j in range(0,GT_data.shape[1]):
  for k in range(0,GT_data.shape[2]):
     k1 = int(round(float(dim3*k)/float(GT_data.shape[2])))
     SegMapT2Space[i,j,k] = SegMap[i,j,k1]


#nclasses = 4
#class_ids = np.array(np.zeros(nclasses),dtype=np.int64)
#class_ids[0] = 85
#class_ids[1] = 50
#class_ids[2] = 190
#class_ids[3] = 255
##   Evaluate
#evaluation.DICE(SegMapT2Space, np.array(GT_data,dtype=np.int64), \
#         ['CGM','SGM','UWM','CSF'], class_ids, nclasses, \
#         results_dir+output_textfile1, 1, final_time)
#evaluation.Accuracy(SegMapT2Space, np.array(GT_data,dtype=np.int64), \
#         ['CGM','SGM','UWM','CSF'], class_ids, nclasses, \
#         results_dir+output_textfile2, 1, final_time)


#   save results
Im = Image.fromarray(np.uint8(SegMapT2Space[:,:,nSliceGT]))
Im.save(results_dir+'experiments/'+output_SegFile+'.png')
Im = Image.fromarray(np.uint8(SegMapT2Space[:,:,nSliceGT+difSliceGT]))
Im.save(results_dir+'experiments/'+output_SegFile+'_p'+str(difSliceGT)+'.png')
Im = Image.fromarray(np.uint8(SegMapT2Space[:,:,nSliceGT-difSliceGT]))
Im.save(results_dir+'experiments/'+output_SegFile+'_m'+str(difSliceGT)+'.png')
SegMapVolume = nib.Nifti1Image(SegMapT2Space,t2CSAffine)
nib.save(SegMapVolume,t2CurrentSubjectName[0:(t2CurrentSubjectName.index("/T2"))]+"/"+output_SegFile+".nii.gz")

