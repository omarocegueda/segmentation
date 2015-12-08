import numpy as np
import nibabel as nib
import os
import pickle
from scipy.ndimage.morphology import binary_dilation

from dipy.align.imwarp import (SymmetricDiffeomorphicRegistration)
from dipy.align.metrics import (CCMetric)
from dipy.align import VerbosityLevels
from dipy.align.imaffine import (AffineMap,
                                 transform_centers_of_mass)
import dipy.viz.regtools as rt
from dataset_info import info
from reg_utils.registration import dipy_align


# Test with adult brains
# Load ibsr
ibsr1_name = info.get_ibsr(1, 'raw')
ibsr1_nib = nib.load(ibsr1_name)
ibsr1 = ibsr1_nib.get_data().squeeze()
ibsr1_affine = ibsr1_nib.get_affine()

# Load brainweb (and match ibsr axes)
brainweb_strip_name = info.get_brainweb('t1','strip')
brainweb_strip_nib = nib.load(brainweb_strip_name)
brainweb_strip = brainweb_strip_nib.get_data().squeeze()
brainweb_strip = brainweb_strip.transpose([0,2,1])[::-1,:,:]
brainweb_mask = brainweb_strip>0

brainweb_name = info.get_brainweb('t1','raw')
brainweb_nib = nib.load(brainweb_name)
brainweb = brainweb_nib.get_data().squeeze()
brainweb_affine = brainweb_nib.get_affine()
brainweb = brainweb.transpose([0,2,1])[::-1,:,:]
rt.plot_slices(brainweb)
brainweb_affine = ibsr1_affine.copy()
brainweb_affine[brainweb_affine!=0] = 1
brainweb_affine[0,0] = -1



# Reslice Brainweb on IBSR1
ibsr_to_bw = AffineMap(None, ibsr1.shape, ibsr1_affine, brainweb.shape, brainweb_affine)
bw_on_ibsr1 = ibsr_to_bw.transform(brainweb)
rt.overlay_slices(ibsr1, bw_on_ibsr1) # misaligned

c_of_mass = transform_centers_of_mass(ibsr1, ibsr1_affine, brainweb, brainweb_affine)
bw_on_ibsr1 = c_of_mass.transform(brainweb)
rt.overlay_slices(ibsr1, bw_on_ibsr1) # roughly aligned

# Start affine alignment
aff_name = 'ibsr1_to_brainweb.p'
if os.path.isfile(aff_name):
    ibsr_bw_affmap = pickle.load(open(aff_name,'r'))
else:
    ibsr_bw_affmap = dipy_align(ibsr1, ibsr1_affine, brainweb, brainweb_affine)
    pickle.dump(ibsr_bw_affmap, open(aff_name,'w'))
bw_on_ibsr1 = ibsr_bw_affmap.transform(brainweb)
rt.overlay_slices(ibsr1, bw_on_ibsr1, slice_type=0) # aligned (sagital view)
rt.overlay_slices(ibsr1, bw_on_ibsr1, slice_type=1) # aligned (axial view)
rt.overlay_slices(ibsr1, bw_on_ibsr1, slice_type=2) # aligned (coronal view)

# Start diffeomorphic registration
diff_name = 'ibsr1_to_brainweb_diff.p'
if os.path.isfile(diff_name):
    ibsr_bw_diffmap = pickle.load(open(diff_name,'r'))
else:
    metric = CCMetric(3)
    level_iters = [50, 10]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    sdr.verbosity = VerbosityLevels.DEBUG
    ibsr_bw_diffmap = sdr.optimize(ibsr1, brainweb, ibsr1_affine, brainweb_affine, ibsr_bw_affmap.affine)
    pickle.dump(ibsr_bw_diffmap, open(diff_name,'w'))

bw_warped_ibsr1 = ibsr_bw_diffmap.transform(brainweb)
rt.overlay_slices(ibsr1, bw_warped_ibsr1, slice_type=0) # warped (sagital view)
rt.overlay_slices(ibsr1, bw_warped_ibsr1, slice_type=1) # warped (axial view)
rt.overlay_slices(ibsr1, bw_warped_ibsr1, slice_type=2) # warped (coronal view)

# Now the initial segmentation
bw_mask_ibsr1 = ibsr_bw_diffmap.transform(brainweb_mask)
bw_mask_ibsr1 = bw_mask_ibsr1>0

# Dilate
structure = np.ones((5,5,5))
dilated_mask = binary_dilation(bw_mask_ibsr1, structure)

ibsr1_strip_init = ibsr1 * dilated_mask
rt.plot_slices(ibsr1_strip_init)

rt.overlay_slices(ibsr1, ibsr1_strip_init, slice_type=0) # warped (sagital view)
rt.overlay_slices(ibsr1, ibsr1_strip_init, slice_type=1) # warped (axial view)
rt.overlay_slices(ibsr1, ibsr1_strip_init, slice_type=2) # warped (coronal view)






