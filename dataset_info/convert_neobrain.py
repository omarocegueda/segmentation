# Standard modules
import numpy as np
import nibabel as nib
import dipy.viz.regtools as rt
from dipy.align.imaffine import (AffineMap,
                                 transform_centers_of_mass)

# Our modules
from dataset_info.info import get_neobrain
from reg_utils.registration import dipy_align

def get_permutation_for_ras(orientation, shape):
    T = np.zeros((4, 4))
    T[3,3] = 1
    axis = {'R':1, 'L':-1, 'A':2, 'P':-2, 'S':3, 'I':-3}
    slices = []
    indices = []
    for i in range(3):
        flag = int(axis[orientation[i]])
        abs_flag = -flag if (flag < 0) else flag
        sign = flag // abs_flag
        idx = abs_flag - 1
        T[idx][i] = sign
        
        # Compute indices and slices to reslice the data
        indices.append(idx)
        if sign < 0:
            slices.append(slice(shape[idx]-1, -(shape[idx] + 1), -1))
        else:
            slices.append(slice(0, shape[idx]))        

    return T, indices, slices

def convert_to_ras(data, affine, orientation):
    r""" Convert affine transform from `orientation` to RAS
    """
    T, indices, slices = get_permutation_for_ras(orientation, data.shape)
    new_data = data.transpose(indices[0], indices[1], indices[2])[slices[0], slices[1], slices[2]]
    print('indices:', indices, 'shape:', data.shape, 'slices:', slices)
    return new_data, affine.dot(T)


def get_header_data(fname):
    r""" Parses a header file and returns the image's geometric information
    """
    D = {}
    with open(fname,'r') as f:
        for line in f.readlines():
            p = line.find('=')
            key = line[:p].strip()
            value = line[(p+1):].strip()
            D[key] = value
    
    # Verify assumptions
    if (D['CenterOfRotation'] != '0 0 0'):
        raise ValueError('Found inconsistent center of rotation')
    if D['CompressedData'] != 'False':
        raise ValueError('Found compressed data')
    if D['BinaryData'] != 'True':
        raise ValueError('Found non-binary data')
    if D['NDims'] != '3':
        raise ValueError('NDims != 3 : %s'(D['NDims'],) )
    if D['ObjectType'] != 'Image':
        raise ValueError('Fojnd non-image file')
            
    # Extract relevant information
    shape = tuple([int(s) for s in D['DimSize'].strip().split()])
    dtype = None
    if D['ElementType'] == 'MET_UCHAR':
        dtype = np.uint8
    elif D['ElementType'] == 'MET_SHORT':
        dtype = np.int16
    elif D['ElementType'] == 'MET_USHORT':
        dtype = np.uint16
    else:
        raise ValueError('Unexpected data type: %s'%(D['ElementType'],))
    
    orientation = D['AnatomicalOrientation']
    spacing = np.array([float(s) for s in D['ElementSpacing'].strip().split()]).astype(np.float64)
    affine_data = np.array([float(s) for s in D['TransformMatrix'].strip().split()]).astype(np.float64)
    offset = np.array([float(s) for s in D['Offset'].strip().split()]).astype(np.float64)
    affine = np.eye(4)
    
    # Warning: we don't know in what order arethe matrix entries given,
    # we assume Fortran because the binary data is Fortran
    #affine[:3,:3] = affine_data.reshape((3,3), order='F')
    affine[:3,:3] = affine_data.reshape((3,3))
    affine[:3,3] = offset
    return shape, affine, spacing, dtype, orientation


def load_from_raw(fname_noext):
    r""" Loads volume from header and binary files
    
    Adds file extension '.mhd' and '.raw' to determine the appropriate
    file names
    """
    if fname_noext[-4:]=='.nii':
        fname_noext = fname_noext[:-4]
        
    hdr = fname_noext + '.mhd'
    raw = fname_noext + '.raw'
    shape, affine, spacing, dtype, orientation = get_header_data(hdr)
    data = np.fromfile(raw, dtype=dtype, count=-1, sep="")
    data = data.reshape(shape, order='F')
    return data, affine, spacing, orientation
    
def convert_neobrain_to_nifti(fname_noext):
    print('Converting %s to Nifti...'%(fname_noext, ))
    neo, neo_affine, neo_spacing, neo_orientation =  load_from_raw(fname_noext)
    new_neo, new_neo_affine = convert_to_ras(neo, neo_affine, neo_orientation)
    neo_nib = nib.Nifti1Image(new_neo, new_neo_affine)
    neo_nib.to_filename(fname_noext + '.nii')

def convert_all():
    # Convert training data
    for idx in range(1, 1 + 4):
        for modality in ['T1', 'T2', 'seg']:
            fname = get_neobrain('train', idx, modality)
            fname_noext = fname[:-4]
            convert_neobrain_to_nifti(fname_noext)
            
    # Convert testing data
    for idx in range(1, 1 + 3):
        for prefix in ['i1', 'i2', 'i3', 'iC1', 'iC2']:
            for suffix in ['t1', 't2']:
                modality = prefix + '_' + suffix
                fname = get_neobrain('test', idx, modality)
                fname_noext = fname[:-4]
                convert_neobrain_to_nifti(fname_noext)

def quick_check():
    # Verify that original and RAS versions of neo1 describe the same object

    # Load original data
    neo1_fname = get_neobrain('train', 1, 'T1')
    neo1_old, neo1_old_affine, neo1_old_spacing, neo1_old_ori = load_from_raw(neo1_fname)
    
    # Load RAS version
    neo1_nib = nib.load(neo1_fname)
    neo1 = neo1_nib.get_data()    
    neo1_affine = neo1_nib.get_affine()
    
    # Resample RAS on top of original
    aff = AffineMap(None, neo1_old.shape, neo1_old_affine, neo1.shape, neo1_affine)
    neo1_resampled = aff.transform(neo1)
    rt.overlay_slices(neo1_old, neo1_resampled, slice_type=0)
    rt.overlay_slices(neo1_old, neo1_resampled, slice_type=1)
    rt.overlay_slices(neo1_old, neo1_resampled, slice_type=2)
    
   
    # Attempt to resample a test volume on top of training
    neo2_fname = get_neobrain('test', 1, 'i1_t1')
    neo2_nib = nib.load(neo2_fname)
    neo2 = neo2_nib.get_data()   
    neo2_affine = neo2_nib.get_affine()
    aff = transform_centers_of_mass(neo1, neo1_affine, neo2, neo2_affine)
    #aff = dipy_align(neo1, neo1_affine, neo2, neo2_affine)
    neo2_resampled = aff.transform(neo2)
    
    rt.overlay_slices(neo1, neo2_resampled, slice_type=0)
    rt.overlay_slices(neo1, neo2_resampled, slice_type=1)
    rt.overlay_slices(neo1, neo2_resampled, slice_type=2)
   
    

    
    
        