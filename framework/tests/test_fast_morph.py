from framework.fast_morph import (SequencialSphereDilation,
                                  create_sphere,
                                  get_list,
                                  get_subsphere_lists,
                                  isotropic_erosion,
                                  isotropic_dilation)
from dipy.align.vector_fields import create_sphere
import dipy.viz.regtools as rt
from skimage.morphology import erosion, dilation, closing, cube, ball
import time
import gc
import nibabel as nib
from dipy.align.reslice import reslice
from numpy.testing import (assert_array_equal)

def test_performance():
    ''' Compare execution time against scikit, sequencial closing case
    '''
    base_dir = '/home/omar/data/DATA_NeoBrainS12/'
    neo_subject = '30wCoronal/example2/'

    # Read subject files
    t2CurrentSubjectName  = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'T2_1-1.nii.gz'
    t2CurrentSubject_data = nib.load(t2CurrentSubjectName).get_data()
    affineT2CS            = nib.load(t2CurrentSubjectName).get_affine()
    zoomsT2CS             = nib.load(t2CurrentSubjectName).get_header().get_zooms()[:3]
    # Step 1.4 - Resampling for isotropic voxels

    n_zooms = (zoomsT2CS[0],zoomsT2CS[0],zoomsT2CS[0])
    t2CurrentSubject_data,affineT2CS = reslice(t2CurrentSubject_data,affineT2CS,zoomsT2CS,n_zooms)

    S = t2CurrentSubject_data.astype(np.float64)
    S = S[:S.shape[0]//4, :S.shape[1]//4, :S.shape[2]//4]

    ###########compare times#########
    # in-house
    start = time.time()
    max_radius = 11
    D = SequencialSphereDilation(S)
    for r in range(max_radius):
        print('Computing radius %d...'%(r+1,))
        D.expand(S)
        actual = D.get_current_closing()
        del actual
    del D
    end = time.time()
    print('Elapsed (in-home): %f'%(end-start,))
    # scikit
    start = time.time()
    for r in range(max_radius):
        print('Computing radius %d...'%(1+r,))
        expected = closing(S, ball(1+r))
        del expected
    end = time.time()
    print('Elapsed (scikit): %f'%(end-start,))


def test_accuracy():
    ''' Verify that our implementation returns exactly the same as scikit
    '''
    base_dir = '/home/omar/data/DATA_NeoBrainS12/'
    neo_subject = '30wCoronal/example2/'

    # Read subject files
    t2CurrentSubjectName  = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'T2_1-1.nii.gz'
    t2CurrentSubject_data = nib.load(t2CurrentSubjectName).get_data()
    affineT2CS            = nib.load(t2CurrentSubjectName).get_affine()
    zoomsT2CS             = nib.load(t2CurrentSubjectName).get_header().get_zooms()[:3]

    n_zooms = (zoomsT2CS[0],zoomsT2CS[0],zoomsT2CS[0])
    t2CurrentSubject_data,affineT2CS = reslice(t2CurrentSubject_data,affineT2CS,zoomsT2CS,n_zooms)

    S = t2CurrentSubject_data.astype(np.float64)

    max_radius = 4
    D = SequencialSphereDilation(S)
    for r in range(1, 1+max_radius):
        D.expand(S)
        expected = dilation(S, ball(r))
        actual = D.get_current_dilation()
        assert_array_equal(expected, actual)
        expected = closing(S, ball(r))
        actual = D.get_current_closing()
        assert_array_equal(expected, actual)


def test_large_radius():
    ''' Compare execution time against scikit: single closing case
    Here, our implementation does not take advantage of smaller radius results
    so ours is slower than scikit, but it uses significantly less memory.
    '''
    base_dir = '/home/omar/data/DATA_NeoBrainS12/'
    neo_subject = '30wCoronal/example2/'

    # Read subject files
    t2CurrentSubjectName  = base_dir + 'trainingDataNeoBrainS12/'+neo_subject+'T2_1-1.nii.gz'
    t2CurrentSubject_data = nib.load(t2CurrentSubjectName).get_data()
    affineT2CS            = nib.load(t2CurrentSubjectName).get_affine()
    zoomsT2CS             = nib.load(t2CurrentSubjectName).get_header().get_zooms()[:3]
    # Step 1.4 - Resampling for isotropic voxels

    n_zooms = (zoomsT2CS[0],zoomsT2CS[0],zoomsT2CS[0])
    t2CurrentSubject_data,affineT2CS = reslice(t2CurrentSubject_data,affineT2CS,zoomsT2CS,n_zooms)

    S = t2CurrentSubject_data.astype(np.float64)

    ###########compare times#########
    # in-house
    radius = 15
    start = time.time()
    d = isotropic_dilation(S, radius)
    c = isotropic_erosion(d, radius)
    end = time.time()
    print('Elapsed (in-home): %f'%(end-start,))

    # scikit
    start = time.time()
    expected = closing(S, ball(radius))
    end = time.time()
    print('Elapsed (scikit): %f'%(end-start,))