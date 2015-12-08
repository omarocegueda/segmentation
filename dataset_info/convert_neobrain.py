# Standard modules
import os
import pickle
import numpy as np
import nibabel as nib
import dipy.viz.regtools as rt
from dipy.align.metrics import CCMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align import VerbosityLevels
from dipy.align.transforms import regtransforms
from dipy.align.imaffine import (AffineMap,
                                 transform_centers_of_mass,
                                 MutualInformationMetric,
                                 LocalCCMetric,
                                 AffineRegistration)
# Our modules
from dataset_info.info import get_neobrain

#from reg_utils.registration import dipy_align

def get_permutation(from_ori, to_ori):
    opposite = {'L':'R', 'R':'L', 'A':'P', 'P':'A', 'I':'S', 'S':'I'}
    axis = {}
    for i in range(3):
        axis[to_ori[i]] = i + 1
        axis[opposite[to_ori[i]]] = - (i + 1)

    T = np.zeros((4, 4))
    T[3,3] = 1
    for i in range(3):
        flag = int(axis[from_ori[i]])
        abs_flag = -flag if (flag < 0) else flag
        sign = flag // abs_flag
        idx = abs_flag - 1
        T[idx][i] = sign
    return T

def convert_affine(affine, from_ori, to_ori):
    r""" Changes the orientation of a grid-to-world transform
    """
    T = get_permutation(from_ori, to_ori)
    return T.dot(affine)


def convert_to_ras(affine, orientation):
    r""" Convert affine transform from `orientation` to RAS
    """
    T = get_permutation(orientation, 'RAS')
    return T.dot(affine)


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
    affine[:3,:3] = affine_data.reshape((3,3), order='F')
    #affine[:3,:3] = affine_data.reshape((3,3))
    affine[:3,3] = offset
    scale = np.eye(4)
    scale[:3,:3] = np.diag(spacing)
    affine = affine.dot(scale)

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
    new_neo_affine = convert_to_ras(neo_affine, neo_orientation)
    #neo_nib = nib.Nifti1Image(neo, new_neo_affine)
    neo_nib = nib.Nifti1Image(neo, new_neo_affine)
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



def quick_check():

    img1_fname = "/home/omar/data/DATA_NeoBrainS12/T1.nii.gz"
    img2_fname = "/home/omar/data/DATA_NeoBrainS12/set2_i1_t1.nii.gz"

    img1_nib = nib.load(img1_fname)
    img1 = img1_nib.get_data().squeeze()
    img1_affine = img1_nib.get_affine()

    img2_nib = nib.load(img2_fname)
    img2 = img2_nib.get_data().squeeze()
    img2_affine = img2_nib.get_affine()
    # nib.aff2axcodes(img1_affine)
    #aff = AffineMap(None, img1.shape, img1_affine, img2.shape, img2_affine)
    #aff = transform_centers_of_mass(img1, img1_affine, img2, img2_affine)
    aff = dipy_align(img1, img1_affine, img2, img2_affine, np.eye(4))

    img2_resampled = aff.transform(img2)
    rt.overlay_slices(img1, img2_resampled, slice_type=0)
    rt.overlay_slices(img1, img2_resampled, slice_type=1)
    rt.overlay_slices(img1, img2_resampled, slice_type=2)



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



    # Load atlas
    atlas_fname = get_neobrain('atlas', 'neo-withSkull', None)
    atlas_nib = nib.load(atlas_fname)
    atlas_affine = atlas_nib.get_affine()
    atlas = atlas_nib.get_data()
    rt.plot_slices(atlas)

    # Resample atlas on top of neo1
    aff = AffineMap(None, neo1.shape, neo1_affine, atlas.shape, atlas_affine)
    atlas_resampled = aff.transform(atlas)
    rt.overlay_slices(neo1, atlas_resampled)


def align_atlas():
    neo_fname = get_neobrain('train', 1, 'T2')
    neo_nib = nib.load(neo_fname)
    neo = neo_nib.get_data()
    neo_affine = neo_nib.get_affine()

    # Load atlas (with skull)
    atlas_fname = get_neobrain('atlas', 'neo-withSkull', None)
    atlas_nib = nib.load(atlas_fname)
    atlas = atlas_nib.get_data()
    atlas_affine = atlas_nib.get_affine()

    # The first training volume dimensions are about  5cm x 5cm x 8cm
    # The atlas  dimensions are about 7cm x 9cm x 11 cm
    # Assuming isotropic scale, the atlas is about 1.5 times larger than
    # the input image:
    iso_scale = (float(7*9*11)/float(5*5*8))**(1.0/3)
    print(iso_scale)

    #We can use this to constraint the transformation to rigid
    scale = np.eye(4)
    scale[:3,:3] *= iso_scale

    rigid_map_fname = 'atlas_towards_neo1_rigid.p'

    if os.path.isfile(rigid_map_fname):
        rigid_map = pickle.load(open(rigid_map_fname, 'r'))
    else:
        transforms = ['RIGID']
        rigid_map = dipy_align(neo, neo_affine, atlas, atlas_affine,
                         transforms=transforms, prealign=scale)
        pickle.dump(rigid_map, open(rigid_map_fname, 'w'))

    atlas_resampled = rigid_map.transform(atlas)

    # Compare anterior coronal slices
    rt.overlay_slices(neo, atlas_resampled, slice_type=2, slice_index=10, ltitle='Neo1', rtitle='Atlas');
    # Compare middle coronal slices
    rt.overlay_slices(neo, atlas_resampled, slice_type=2, slice_index=25, ltitle='Neo1', rtitle='Atlas');
    # Compare posterior coronal slices
    rt.overlay_slices(neo, atlas_resampled, slice_type=2, slice_index=40, ltitle='Neo1', rtitle='Atlas');

    # Load the peeled atlas
    atlas_wcerebellum_fname = get_neobrain('atlas', 'neo-withCerebellum', None)
    atlas_wcerebellum_nib = nib.load(atlas_wcerebellum_fname)
    atlas_wcerebellum = atlas_wcerebellum_nib.get_data()
    atlas_wcerebellum_affine = atlas_wcerebellum_nib.get_affine()

    # Configure diffeomorphic registration
    diff_map_name = 'atlas_towards_neo1_diff.p'
    if os.path.isfile(diff_map_name):
        diff_map = pickle.load(open(diff_map_name, 'r'))
    else:
        metric = CCMetric(3)
        sdr = SymmetricDiffeomorphicRegistration(metric)
        # The atlases are not aligned in physical space!! use atlas_affine instead of atlas_wcerebellum_affine
        diff_map = sdr.optimize(neo, atlas_wcerebellum, neo_affine, atlas_affine, prealign=rigid_map.affine)
        pickle.dump(diff_map, open(diff_map_name, 'w'))

    atlas_wcerebellum_deformed = diff_map.transform(atlas_wcerebellum)

    # Before and after diffeomorphic registration
    rt.overlay_slices(neo, atlas_resampled, slice_type=2, slice_index=10, ltitle='Neo1', rtitle='Atlas');
    rt.overlay_slices(neo, atlas_wcerebellum_deformed, slice_type=2, slice_index=10, ltitle='Neo1', rtitle='Atlas');
    # Before and after diffeomorphic registration
    rt.overlay_slices(neo, atlas_resampled, slice_type=2, slice_index=25, ltitle='Neo1', rtitle='Atlas');
    rt.overlay_slices(neo, atlas_wcerebellum_deformed, slice_type=2, slice_index=25, ltitle='Neo1', rtitle='Atlas');
    # Before and after diffeomorphic registration
    rt.overlay_slices(neo, atlas_resampled, slice_type=2, slice_index=40, ltitle='Neo1', rtitle='Atlas');
    rt.overlay_slices(neo, atlas_wcerebellum_deformed, slice_type=2, slice_index=40, ltitle='Neo1', rtitle='Atlas');
    
    
    
    
    # Now all volumes
    atlas_fname = get_neobrain('atlas', 'neo-withSkull', None)
    atlas_nib = nib.load(atlas_fname)
    atlas = atlas_nib.get_data()
    atlas_affine = atlas_nib.get_affine()
    
    atlas_wcerebellum_fname = get_neobrain('atlas', 'neo-withCerebellum', None)
    atlas_wcerebellum_nib = nib.load(atlas_wcerebellum_fname)
    atlas_wcerebellum = atlas_wcerebellum_nib.get_data()
    atlas_wcerebellum_affine = atlas_wcerebellum_nib.get_affine()
    
    
    
    
    
    idx = 2
    neoi_fname = get_neobrain('train', idx, 'T2')
    neoi_nib = nib.load(neoi_fname)
    neoi = neoi_nib.get_data()
    neoi_affine = neoi_nib.get_affine()
    
    iso_scale = (float(7*9*11)/float(5*5*8))**(1.0/3)
    print(iso_scale)
    

    #We can use this to constraint the transformation to rigid
    scale = np.eye(4)
    scale[:3,:3] *= iso_scale

    rigid_map_fname = 'atlas_towards_neo%d_affine.p'%(idx,)

    if os.path.isfile(rigid_map_fname):
        rigid_map = pickle.load(open(rigid_map_fname, 'r'))
    else:
        transforms = ['RIGID', 'AFFINE']
        level_iters = [[10000, 1000, 100], [100]]
        rigid_map = dipy_align(neoi, neoi_affine, atlas, atlas_affine,
                               transforms=transforms,
                               level_iters=level_iters,
                               prealign=scale)
        pickle.dump(rigid_map, open(rigid_map_fname, 'w'))

    atlas_resampled = rigid_map.transform(atlas)
    rt.overlay_slices(neoi, atlas_resampled, slice_type=2, slice_index = 6)
    rt.overlay_slices(neoi, atlas_resampled, slice_type=2, slice_index = 10)
    rt.overlay_slices(neoi, atlas_resampled, slice_type=2, slice_index = 25)
    rt.overlay_slices(neoi, atlas_resampled, slice_type=2, slice_index = 40)
    
    diff_map_name = 'atlas_towards_neo%d_diff.p'%(idx,)
    if os.path.isfile(diff_map_name):
        diff_map = pickle.load(open(diff_map_name, 'r'))
    else:
        metric = CCMetric(3)
        sdr = SymmetricDiffeomorphicRegistration(metric)
        # The atlases are not aligned in physical space!! use atlas_affine instead of atlas_wcerebellum_affine
        diff_map = sdr.optimize(neoi, atlas_wcerebellum, neoi_affine, atlas_affine, prealign=rigid_map.affine)
        pickle.dump(diff_map, open(diff_map_name, 'w'))

    atlas_wcerebellum_deformed = diff_map.transform(atlas_wcerebellum)

    # Before and after diffeomorphic registration
    rt.overlay_slices(neoi, atlas_resampled, slice_type=2, slice_index=6, ltitle='Neo%d'%(idx,), rtitle='Atlas');
    rt.overlay_slices(neoi, atlas_wcerebellum_deformed, slice_type=2, slice_index=6, ltitle='Neo%d'%(idx,), rtitle='Atlas');
    
    rt.overlay_slices(neoi, atlas_resampled, slice_type=2, slice_index=10, ltitle='Neo%d'%(idx,), rtitle='Atlas');
    rt.overlay_slices(neoi, atlas_wcerebellum_deformed, slice_type=2, slice_index=10, ltitle='Neo%d'%(idx,), rtitle='Atlas');
    # Before and after diffeomorphic registration
    rt.overlay_slices(neoi, atlas_resampled, slice_type=2, slice_index=25, ltitle='Neo%d'%(idx,), rtitle='Atlas');
    rt.overlay_slices(neoi, atlas_wcerebellum_deformed, slice_type=2, slice_index=25, ltitle='Neo%d'%(idx,), rtitle='Atlas');
    # Before and after diffeomorphic registration
    rt.overlay_slices(neoi, atlas_resampled, slice_type=2, slice_index=40, ltitle='Neo%d'%(idx,), rtitle='Atlas');
    rt.overlay_slices(neoi, atlas_wcerebellum_deformed, slice_type=2, slice_index=40, ltitle='Neo%d'%(idx,), rtitle='Atlas');
    
    
