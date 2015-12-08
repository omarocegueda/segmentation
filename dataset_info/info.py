import os
_ibsr_base_dir = 'Unspecified'
_lpba_base_dir = 'Unspecified'
_brainweb_base_dir = 'Unspecified'
_scil_base_dir = 'Unspecified'
_neobrain_base_dir = 'Unspecified'

def getBaseFileName(fname):
    base=os.path.basename(fname)
    noExt=os.path.splitext(base)[0]
    while(noExt!=base):
        base=noExt
        noExt=os.path.splitext(base)[0]
    return noExt


def decompose_path(fname):
    dirname=os.path.dirname(fname)
    if len(dirname)>0:
        dirname += '/'

    base=os.path.basename(fname)
    no_ext = os.path.splitext(base)[0]
    while(no_ext !=base):
        base=no_ext
        no_ext =os.path.splitext(base)[0]
    ext = os.path.basename(fname)[len(no_ext):]
    return dirname, base, ext

def _load_dataset_info():
    r""" Loads the dataset location in the local file system

    There must be a text file '.../segmentation/dataset_info/info.txt'
    containing the location of the five common datasets:
    1. IBSR18
    2. LPBA40
    3. Brainweb
    4. SCIL
    5. Neobrain

    For example:
        /home/myusername/data/IBSR/
        /home/myusername/data/LPBA/
        /home/myusername/data/Brainweb/
        /home/myusername/data/SCIL/
        /home/myusername/data/DATA_NeoBrainS12/

    this module expects exactly 5 nonempty lines, if any of the datasets is not
    available, please add an 'dummy' path

    """
    dirname, base, ext = decompose_path(__file__)
    fname = dirname + base + '.txt'
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = [s.strip() for s in f.readlines()]
            if len(lines) != 5:
                print('Warning: expected base directories for IBSR, LPBA, Brainweb, SCIL and Neobrain in '+fname+' in that order. Found '+str(len(lines))+' lines in file, you may get unexpected results')
            else:
                global _ibsr_base_dir
                global _lpba_base_dir
                global _brainweb_base_dir
                global _scil_base_dir
                global _neobrain_base_dir
                _ibsr_base_dir = lines[0]
                _lpba_base_dir = lines[1]
                _brainweb_base_dir = lines[2]
                _scil_base_dir = lines[3]
                _neobrain_base_dir = lines[4]
    else:
        print('Error: file not found. Expected base directories for IBSR, LPBA, Brainweb, SCIL and Neobrain in text file "'+fname+'" in that order.')

_load_dataset_info()


def get_ibsr_base_dir():
    global _ibsr_base_dir
    return _ibsr_base_dir


def get_lpba_base_dir():
    global _lpba_base_dir
    return _lpba_base_dir


def get_brainweb_base_dir():
    global _brainweb_base_dir
    return _brainweb_base_dir


def get_scil_base_dir():
    global _scil_base_dir
    return _scil_base_dir

def get_neobrain_base_dir():
    global _neobrain_base_dir
    return _neobrain_base_dir

def get_ibsr(idx, data):
    ibsr_base_dir = get_ibsr_base_dir()
    if idx<10:
        idx = '0'+str(idx)
    else:
        idx = str(idx)
    prefix = ibsr_base_dir + 'IBSR_'+idx+'/IBSR_'+idx
    fname = None
    if data == 'mask':
        fname = prefix + '_ana_brainmask.nii.gz'
    elif data == 'seg3':
        fname = prefix + '_segTRI_fill_ana.nii.gz'
    elif data == 'seg':
        fname = prefix + '_seg_ana.nii.gz'
    elif data == 'raw':
        fname = prefix + '_ana.nii.gz'
    elif data == 'strip':
        fname = prefix + '_ana_strip.nii.gz'
    elif data == 't1':
        fname = prefix + '_ana_strip.nii.gz'
    elif data == 't2':
        fname = prefix + '_ana_strip_t2.nii.gz'
    return fname


def get_lpba(idx, data):
    lpba_base_dir = get_lpba_base_dir()
    if idx<10:
        idx = '0'+str(idx)
    else:
        idx = str(idx)
    prefix = lpba_base_dir + 'S'+idx+'/S'+idx
    fname = None
    if data == 'seg':
        fname = prefix + '_seg.img'
    elif data == 'strip':
        fname = prefix + '_strip.img'
    elif data == 'strip_seg':
        fname = prefix + '_strip_seg.img'
    return fname


def get_brainweb(modality, data):
    if not modality in ['t1', 't2', 'pd']:
        return None
    modality = modality.lower()
    brainweb_dir = get_brainweb_base_dir()+modality
    fname = None
    if data is 'strip':
        fname = brainweb_dir+'/brainweb_'+modality+'_strip.nii.gz'
    elif data is 'raw':
        fname = brainweb_dir+'/brainweb_'+modality+'.nii.gz'
    return fname


def get_scil(idx, data):
    scil_base_dir = get_scil_base_dir()
    if idx<10:
        idx = '0'+str(idx)
    else:
        idx = str(idx)
    prefix = scil_base_dir + 'SCIL_' + idx + '/SCIL_' + idx + '_'
    fname = prefix + data + '.nii.gz'
    return fname


def get_neobrain(subset, vol_id, modality):
    r"""
    Parameters
    ----------
    subset : string
        any of 'atlas', 'train' or 'test'
    vol_id : integer or string, depending on `subset`
        if `subset` == 'atas':
            `vol_id` must identify the volume:
                'neo'
                'neo-aal'
                'neo-avgseg'
                'neo-seg'
                'neo-seg-csf'
                'neo-seg-gm'
                'neo-seg-wm'
                'neo-withCerebellum'
                'neo-withSkull'
        if `subset` == 'train':
            `vol_id` must be an int in [1,4]
        if `subset` == 'test':
            `vol_id` must be an int in [1,3]
    modality : string
        if `subset` == 'atlas':
            `modality` is ignored
        if `subset` == 'train':
            `modality` must be any of 'T1', 'T2', 'seg'
        if `subset` == 'test':
            `modality` must be the suffix of the requested file, it consists
            of two strings separated by underscore as follows:
            [i1|i2|i3|iC1|iC2]_[t1|t2]
    """
    neobrain_base_dir = get_neobrain_base_dir()
    if subset == 'train':
        if modality == 'T1' or modality == 'T2':
            prefix = modality
        elif modality == 'seg':
            prefix = 'manualSegm'
        else:
            raise ValueError('Unknown training modality.')

        if vol_id == 1:
            return neobrain_base_dir + 'trainingDataNeoBrainS12/30wCoronal/example1/' + prefix + '.nii.gz'
        elif vol_id == 2:
            return neobrain_base_dir + 'trainingDataNeoBrainS12/30wCoronal/example2/' + prefix + '.nii.gz'
        elif vol_id == 3:
            return neobrain_base_dir + 'trainingDataNeoBrainS12/40wAxial/example3/' + prefix + '.nii.gz'
        elif vol_id == 4:
            return neobrain_base_dir + 'trainingDataNeoBrainS12/40wAxial/example4/' + prefix + '.nii.gz'
        else:
            raise ValueError('Unknown training volume identifier. Expecte integer in [1, 4]')
    elif subset == 'test':
        if isinstance(modality, str):
            if vol_id == 1:
                return neobrain_base_dir + 'scans_axial_40/set1_' + modality + '.nii.gz'
            elif vol_id == 2:
                return neobrain_base_dir + 'scans_cor_30/set2_' + modality + '.nii.gz'
            elif vol_id == 3:
                return neobrain_base_dir + 'scans_cor_40/set3_' + modality + '.nii.gz'
            else:
                raise ValueError('Unknown training volume identifier. Expecte integer in [1, 3]')
        else:
            raise ValueError('Unknown testing modality specifier. Expected string with format [i1|i2|i3|iC1|iC2]_[t1|t2]')
    elif subset == 'atlas':
        if isinstance(vol_id,str):
            return neobrain_base_dir + 'atlas/infant-' + vol_id + '.nii.gz'
        else:
            raise ValueError('Unknown atlas volume identifier.')
    else:
        raise ValueError('Unknown dataset %s'%(subset,))


