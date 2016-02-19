#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from skimage.morphology import ball
cdef extern from "math.h" nogil:
    double floor(double)
    double sqrt(double)

cdef inline double _double_max(double a, double b) nogil:
    r"""
    Returns the maximum of a and b
    """
    return a if a >= b else b


cdef inline double _double_min(double a, double b) nogil:
    r"""
    Returns the minimum of a and b
    """
    return a if a <= b else b

cdef inline int _int_max(int a, int b) nogil:
    r"""
    Returns the maximum of a and b
    """
    return a if a >= b else b


cdef inline int _int_min(int a, int b) nogil:
    r"""
    Returns the minimum of a and b
    """
    return a if a <= b else b

def create_sphere(cnp.npy_intp nslices, cnp.npy_intp nrows,
                  cnp.npy_intp ncols, cnp.npy_intp radius):
    r"""
    Create a binary 3D image where voxel values are 1 iff their distance
    to the center of the image is less than or equal to radius.

    Parameters
    ----------
    nslices : int
        number if slices of the resulting image
    nrows : int
        number of rows of the resulting image
    ncols : int
        number of columns of the resulting image
    radius : int
        the radius of the sphere

    Returns
    -------
    c : array, shape (nslices, nrows, ncols)
        the binary image of the sphere with the requested dimensions
    """
    cdef:
        cnp.npy_intp mid_slice = nslices//2
        cnp.npy_intp mid_row = nrows//2
        cnp.npy_intp mid_col = ncols//2
        cnp.npy_intp i, j, k, ii, jj, kk
        int r2, radius2
        int[:, :, :] s = np.zeros((nslices, nrows, ncols), dtype=np.int32)

    with nogil:
        radius2 = radius*radius
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    kk = k - mid_slice
                    ii = i - mid_row
                    jj = j - mid_col
                    r2 = ii*ii + jj*jj + kk*kk
                    if r2 <= radius2:
                        s[k, i, j] = 1
                    else:
                        s[k, i, j] = 0
    return np.asarray(s)


def get_sphere(int radius):
    cdef:
        int max_side = 1 + 2 * radius
        int max_points, cnt
        int x, y, z, r
        int[:,:,:] prev
        int[:,:,:] current
        int[:,:] out
    prev = create_sphere(max_side, max_side, max_side, radius - 1)
    current = create_sphere(max_side, max_side, max_side, radius)

    with nogil:
        max_points = 0
        for x in range(max_side):
            for y in range(max_side):
                for z in range(max_side):
                    if current[x, y, z] > prev[x, y, z]:
                        max_points += 1
    out = np.zeros((max_points, 3), dtype=np.int32)
    with nogil:
        cnt = 0
        for x in range(max_side):
            for y in range(max_side):
                for z in range(max_side):
                    if current[x, y, z] > prev[x, y, z]:
                        out[cnt, 0] = x - radius
                        out[cnt, 1] = y - radius
                        out[cnt, 2] = z - radius
                        cnt += 1
    return np.asarray(out)


def get_list(int[:,:,:] mask, int radius):
    cdef:
        int nx = mask.shape[0]
        int ny = mask.shape[1]
        int nz = mask.shape[2]
        int cnt
        int x, y, z
        double[:,:,:] prev
        double[:,:,:] current
        int[:,:] out
    with nogil:
        cnt = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if mask[x, y, z] != 0:
                        cnt += 1
    out = np.zeros((cnt, 3), dtype=np.int32)
    with nogil:
        cnt = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if mask[x, y, z] != 0:
                        out[cnt, 0] = x - radius
                        out[cnt, 1] = y - radius
                        out[cnt, 2] = z - radius
                        cnt += 1
    return np.asarray(out)


def get_subsphere_lists(int r, int R):
    ''' Obtains the set of voxels of the radius R sphere that are not covered
    by the set of displaced sub-spheres of radius r

    Return
    ------
        bcenters : array, shape(7, 3)
            array of offsets of sub-sphere centers. Given a point (x, y, z),
            the center of the ith sub-sphere of the structure element centered
            at (x,y,z) is (x+bcenters[i,0], x+bcenters[i,1], x+bcenters[i,2])
        points : array, shape(n, 3)
            array of points within the sphare of radius R not contained in the
            union of the set of 7 sub-spheres. The ith point of the structure
            element of radius R centered at (x, y, z) is
            (x+points[i, 0], y+points[i, 1], z+points[i, 2])
    '''
    cdef:
        int x, y, z, xx, yy, zz, cx, cy, cz, cnt
        int dr = R-r
        int S = 1 + 2 * R
        int s = 1 + 2 * r
        int* dX = [0, -dr, 0, dr,  0, 0, 0]
        int* dY = [0, 0, dr, 0, -dr, 0, 0]
        int* dZ = [0, 0, 0, 0, 0, -dr, dr]
        int nballs = 7
        int[:,:,:] mask = np.zeros((S, S, S), dtype=np.int32)
        int[:,:] points
        int[:,:] bcenters
        int[:,:,:] small_sphere
        int[:,:,:] big_sphere

    if R < r:
        raise ValueError("Small radius must not be bigger than big radius")
    small_sphere = create_sphere(s, s, s, r)
    big_sphere = create_sphere(S, S, S, R)
    bcenters = np.empty((nballs, 3), dtype=np.int32)
    with nogil:
        for k in range(nballs):
            cx = R + dX[k]
            cy = R + dY[k]
            cz = R + dZ[k]
            bcenters[k, 0] = dX[k]
            bcenters[k, 1] = dY[k]
            bcenters[k, 2] = dZ[k]
            for x in range(s):
                xx = cx + (x - r)
                for y in range(s):
                    yy = cy + (y - r)
                    for z in range(s):
                        zz = cz + (z - r)
                        if small_sphere[x, y, z] != 0:
                            mask[xx, yy, zz] = 1
        cnt = 0
        for x in range(S):
            for y in range(S):
                for z in range(S):
                    if big_sphere[x, y, z]>mask[x, y, z]:
                        cnt += 1
    points = np.empty((cnt, 3), dtype=np.int32)
    with nogil:
        cnt = 0
        for x in range(S):
            for y in range(S):
                for z in range(S):
                    if big_sphere[x, y, z]>mask[x, y, z]:
                        points[cnt, 0] = x - R
                        points[cnt, 1] = y - R
                        points[cnt, 2] = z - R
                        cnt += 1
                    elif big_sphere[x, y, z]<mask[x, y, z]:
                        with gil:
                            raise ValueError('Subsphere not contained in main sphere.')
    return np.asarray(bcenters), np.asarray(points)


cdef class SequencialSphereDilation(object):
    cdef:
        double[:,:,:,:] MM
        int p, current_radius
        int nx, ny, nz

    def __init__(self, double[:,:,:] image):
        r""" Fast morphological operations with square structure element.
        """
        self.nx = image.shape[0]
        self.ny = image.shape[1]
        self.nz = image.shape[2]
        self.MM = np.empty((2,self.nx, self.ny, self.nz), dtype=np.float64)
        self.MM[0,...] = image[...]
        self.p = 0
        self.current_radius = 0


    def expand(self, double[:,:,:] image):
        cdef:
            int[:,:] bcenters
            int[:,:] points

            int nballs, npoints
            int i, n, x, y, z, xx, yy, zz
            double v
        bcenters, points = get_subsphere_lists(self.current_radius, 1+self.current_radius)
        self.current_radius += 1
        nballs = bcenters.shape[0]
        npoints = points.shape[0]

        with nogil:
            for x in range(self.nx):
                for y in range(self.ny):
                    for z in range(self.nz):
                        v = self.MM[self.p,x,y,z]  # minimum over small ball centered at (x, y, z)
                        for i in range(nballs):
                            # Evaluate at the center of the ith ball
                            xx = x + bcenters[i, 0]
                            if xx < 0 or xx >= self.nx:
                                continue
                            yy = y + bcenters[i, 1]
                            if yy < 0 or yy >= self.ny:
                                continue
                            zz = z + bcenters[i, 2]
                            if zz < 0 or zz >= self.nz:
                                continue
                            if self.MM[self.p,xx,yy,zz] > v:
                                v = self.MM[self.p,xx,yy,zz]

                        for i in range(npoints):
                            xx = x + points[i, 0]
                            if xx < 0 or xx >= self.nx:
                                continue
                            yy = y + points[i, 1]
                            if yy < 0 or yy >= self.ny:
                                continue
                            zz = z + points[i, 2]
                            if zz < 0 or zz >= self.nz:
                                continue
                            if image[xx, yy, zz] > v:
                                v = image[xx, yy, zz]  # voxel value of the image being eroded
                        self.MM[1-self.p, x, y, z] = v  # result for big radius
            self.p = 1 - self.p

    def get_current_dilation(self):
        return np.asarray(self.MM[self.p])

    def get_current_closing(self):
        return isotropic_erosion(self.MM[self.p], self.current_radius)


def isotropic_erosion(double[:,:,:] image, int radius):
    cdef:
        int nx = image.shape[0]
        int ny = image.shape[1]
        int nz = image.shape[2]
        int x, y, z, xx, yy, zz, r, R, side, i, n, p
        double v
        double[:,:,:,:] M = np.zeros((2, nx, ny, nz))
        int[:,:] bcenters
        int[:,:] points
        int nballs, npoints

    # initialize with current dilation result
    #image = np.ascontiguousarray(image)
    M[0,...] = image[...]
    p = 0
    r = 0
    R = 1
    # R is the big radius resul (no longer computed, it's unavailable)
    # r is the small radius result, available at array position p
    while R <= radius:  # Is it necessary to compute result for R?
        # Compute result for big radius R
        print("[Erosion]Processing radius %d / %d"%(R, radius))
        bcenters, points = get_subsphere_lists(r, R)
        nballs = bcenters.shape[0]
        npoints = points.shape[0]

        with nogil:
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        v = M[p,x,y,z]  # minimum over small ball centered at (x, y, z)
                        for i in range(nballs):
                            # Evaluate at the center of the ith ball
                            xx = x + bcenters[i, 0]
                            if xx < 0 or xx >= nx:
                                continue
                            yy = y + bcenters[i, 1]
                            if yy < 0 or yy >= ny:
                                continue
                            zz = z + bcenters[i, 2]
                            if zz < 0 or zz >= nz:
                                continue
                            if M[p,xx,yy,zz] < v:
                                v = M[p,xx,yy,zz]

                        for i in range(npoints):
                            xx = x + points[i, 0]
                            if xx < 0 or xx >= nx:
                                continue
                            yy = y + points[i, 1]
                            if yy < 0 or yy >= ny:
                                continue
                            zz = z + points[i, 2]
                            if zz < 0 or zz >= nz:
                                continue
                            if image[xx, yy, zz] < v:
                                v = image[xx, yy, zz]
                        M[1-p, x, y, z] = v  # result for big radius R
            p = 1 - p
            r = R  # now this is the last radius available at position p
            if R == radius:
                break
            else:
                R = _int_min(2 * R, radius)
    return np.array(M[p,...])


def isotropic_dilation(double[:,:,:] image, int radius):
    cdef:
        int nx = image.shape[0]
        int ny = image.shape[1]
        int nz = image.shape[2]
        int x, y, z, xx, yy, zz, r, R, side, i, n, p
        double v
        double[:,:,:,:] M = np.zeros((2, nx, ny, nz))
        int[:,:] bcenters
        int[:,:] points
        int nballs, npoints

    # initialize with current dilation result
    #image = np.ascontiguousarray(image)
    M[0,...] = image[...]
    p = 0
    r = 0
    R = 1
    # R is the big radius resul (no longer computed, it's unavailable)
    # r is the small radius result, available at array position p
    while R <= radius:  # Is it necessary to compute result for R?
        print("[Dilation]Processing radius %d / %d"%(R, radius))
        # Compute result for big radius R
        bcenters, points = get_subsphere_lists(r, R)
        nballs = bcenters.shape[0]
        npoints = points.shape[0]

        with nogil:
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        v = M[p,x,y,z]  # minimum over small ball centered at (x, y, z)
                        for i in range(nballs):
                            # Evaluate at the center of the ith ball
                            xx = x + bcenters[i, 0]
                            if xx < 0 or xx >= nx:
                                continue
                            yy = y + bcenters[i, 1]
                            if yy < 0 or yy >= ny:
                                continue
                            zz = z + bcenters[i, 2]
                            if zz < 0 or zz >= nz:
                                continue
                            if M[p,xx,yy,zz] > v:
                                v = M[p,xx,yy,zz]

                        for i in range(npoints):
                            xx = x + points[i, 0]
                            if xx < 0 or xx >= nx:
                                continue
                            yy = y + points[i, 1]
                            if yy < 0 or yy >= ny:
                                continue
                            zz = z + points[i, 2]
                            if zz < 0 or zz >= nz:
                                continue
                            if image[xx, yy, zz] > v:
                                v = image[xx, yy, zz]
                        M[1-p, x, y, z] = v  # result for big radius R
            p = 1 - p
            r = R  # now this is the last radius available at position p
            if R == radius:
                break
            else:
                R = _int_min(2 * R, radius)
    return np.array(M[p,...])

