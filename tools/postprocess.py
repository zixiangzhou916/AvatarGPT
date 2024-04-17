import numpy as np
import numpy.linalg as la
import collections
import typing as t
# import scipy.ndimage as ndim
import scipy
from scipy import interpolate
from scipy.spatial.transform import Slerp
import math
from math import factorial

Param2 = collections.namedtuple('Param2', ('row', 'column'))
Polynom2 = collections.namedtuple('Polynom2', ('row_pows', 'column_pows', 'num_coeffs'))

_Param2Type = t.Union[Param2, t.Tuple[int, int]]
_ParamType = t.Union[int, _Param2Type]

_DIM = 2

def savitzky_golay_1d(y, window_size, order, deriv=0, rate=1):
        
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number") 
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

class SGolayKernel2:
    """Computes two-dimensional kernel (weights) for Savitzky-Golay filter
    """

    def __init__(self, window_size: _Param2Type, poly_order: _Param2Type):
        self._window_size = Param2(*window_size)
        self._poly_order = Param2(*poly_order)

        self._kernel = None  # type: np.ndarray
        self.computed = False

    def __call__(self):
        self.compute()

    def compute(self):
        if self.computed:
            return

        polynom = self._make_polynom(self._poly_order)
        basis_matrix = self._make_basis_matrix(self._window_size, polynom)

        self._kernel = self._compute_kernel(self._window_size, basis_matrix)

        self.computed = True

    @property
    def kernel(self):
        """Returns 2D Savitzky-Golay kernel
        """
        self.compute()
        return self._kernel

    @staticmethod
    def _make_polynom(poly_order: Param2):
        """
        Creates 2-D polynom model (for example poly33):
            p = a00 + a10x + a01y + a20x^2 + a11xy + a02y^2 + a30x^3 + a21x^2y \
                + a12xy^2 + a03y^3
        """
        row_pows = []
        column_pows = []
        num_coeffs = 0

        for row in range(poly_order.row + 1):
            for column in range(poly_order.column + 1):
                if (row + column) > max(*poly_order):
                    continue

                row_pows.append(row)
                column_pows.append(column)

                num_coeffs += 1

        return Polynom2(row_pows, column_pows, num_coeffs)

    @staticmethod
    def _make_basis_matrix(window_size: Param2, poly: Polynom2):
        """Creates basis polynomial matrix
        """
        basis_rows = window_size.row * window_size.column
        basis_columns = poly.num_coeffs

        basis_matrix = np.zeros((basis_rows, basis_columns))

        radius_row = (window_size.row - 1) // 2
        radius_column = (window_size.column - 1) // 2

        row_pows = np.array(poly.row_pows)
        column_pows = np.array(poly.column_pows)

        k = 0

        for row in range(-radius_row, radius_row + 1):
            for column in range(-radius_column, radius_column + 1):
                basis_matrix[k, :] = column ** column_pows * row ** row_pows
                k += 1

        return basis_matrix

    @staticmethod
    def _compute_kernel(window_size: Param2,
                        basis_matrix: np.ndarray):
        """Computes filter 2D kernel via solving least squares problem
        """
        q, _ = la.qr(basis_matrix)

        iq = (window_size.row * window_size.column - 1) // 2
        kernel = q @ np.array(q[iq, :], ndmin=2).T
        kernel = np.fliplr(kernel.reshape(*window_size, order='F'))

        return kernel

class SGolayFilter2(object):
    """Two-dimensional Savitzky-Golay filter
    """

    def __init__(self, window_size: _ParamType, poly_order: _ParamType):
        self._window_size = self._canonize_param(
            'window_size', window_size, self._validate_window_size)
        self._poly_order = self._canonize_param(
            'poly_order', poly_order, self._validate_poly_order)

        self._kernel = SGolayKernel2(self._window_size, self._poly_order)

    def __call__(self, data: np.ndarray,
                 mode: str = 'reflect', cval: float = 0.0):
        return self._filtrate(data, mode=mode, cval=cval)

    @property
    def window_size(self):
        return self._window_size

    @property
    def poly_order(self):
        return self._poly_order

    @property
    def kernel(self):
        """Returns filter 2D kernel
        """
        return self._kernel

    @staticmethod
    def _canonize_param(name, value: _ParamType, validator):
        err = TypeError(
            'The parameter "{}" must be int scalar or Tuple[int, int]'.format(
                name))

        if isinstance(value, int):
            value = (value, value)

        if not isinstance(value, (list, tuple)):
            raise err
        if len(value) != _DIM:
            raise err
        if not all(isinstance(v, int) for v in value):
            raise err

        validator(value)

        return Param2(*value)

    @staticmethod
    def _validate_window_size(value):
        if not all(v >= 3 and bool(v % 2) for v in value):
            raise ValueError(
                'Window size values must be odd and >= 3 (Given: {})'.format(
                    value))

    @staticmethod
    def _validate_poly_order(value):
        if not all(v >= 1 for v in value):
            raise ValueError(
                'Polynom order values must be >= 1 (Given: {})'.format(value))

    def _filtrate(self, data: np.ndarray, *args, **kwargs):
        self._kernel.compute()
        return ndim.correlate(data, self._kernel.kernel, *args, **kwargs)

def smooth_joints(joints, filter_dim="1d"):
    """Smooth the joints sequence using savitzky-golay algorithm
    :param joints: [seq_len, num_joints, 3]
    """
    num_joints = joints.shape[1]
    smoothed_joints = []
    for i in range(num_joints):
        if filter_dim == "1d":
            s_joints = []
            for j in range(3):
                s_joints.append(savitzky_golay_1d(joints[:, i, j], 11, 3))
            s_joints = np.stack(s_joints, axis=-1)
            smoothed_joints.append(s_joints)
        elif filter_dim == "2d":
            s_joints = SGolayFilter2(window_size=11, poly_order=3)(joints[:, i])
            smoothed_joints.append(s_joints)
            
    smoothed_joints = np.stack(smoothed_joints, axis=1)
    return smoothed_joints

def correct_sudden_jittering(joints, threshold=0.1):
    """Correct the sudden jittering of the joints sequence.
    :param joints: [seq_len, num_joints, 3]
    """
    while True:
        
        # Firstly we smooth the joints using savitzky_golay_1d
        smoothed_joints = smooth_joints(joints=joints, filter_dim="1d")
        # Calculate the difference between raw and smoothed joints
        diff_joints = np.linalg.norm(smoothed_joints - joints, axis=-1).max(axis=-1)
        # Get frames of correct
        frame_mask = np.where(diff_joints > threshold)[0]
        if len(frame_mask) == 0:
            break
        else:
            joints = smoothed_joints.copy()
        
    return smoothed_joints

def adaptive_correct_sudden_jittering(joints, threshold=0.1):
    """Correct the sudden jittering of the joints sequence.
    :param joints: [seq_len, num_joints, 3]
    """
    joints_w_origin = joints - joints[:, :1]                    # pose w.r.t origin
    diff_w_prev = joints_w_origin[1:] - joints_w_origin[:-1]    # positional displacement w.r.t previous frame
    diff_w_prev = np.linalg.norm(diff_w_prev, axis=-1).mean(axis=-1)
    diff_w_post = joints_w_origin[:-1] - joints_w_origin[1:]    # positional displacement w.r.t post frame
    diff_w_post = np.linalg.norm(diff_w_post, axis=-1).mean(axis=-1)
    
    jittering_w_prev = np.where(diff_w_prev > threshold)[0] + 1
    jittering_w_post = np.where(diff_w_post > threshold)[0]
    
    jittering = np.concatenate([jittering_w_prev, jittering_w_post], axis=0)
    jittering = np.unique(jittering)
    # jittering = np.array([10,11,12, 20,21,22, 30, 31, 40])
    
    jittering_clusters = []
    if len(jittering) >= 2:
        intervals = jittering[1:] - jittering[:-1]
        gaps = np.where(intervals > 1)[0]
        gaps += 1
        gaps = gaps.tolist()

        if len(gaps) == 0:
            # Only one cluster
            jittering_clusters.append(jittering) 
        else:
            # Multiple cluster
            gaps = [0] + gaps + [len(intervals) + 1]
            for k in range(1, len(gaps)):
                i = gaps[k-1]
                j = gaps[k]
                jittering_clusters.append(jittering[i:j])
    else:
        jittering_clusters.append(jittering)
    
    for jittering in jittering_clusters:
        if len(jittering) == 0: continue
        i = jittering[0] - 1
        j = jittering[-1] + 1
        # print(i, j)
        interp_joints = linear_interpolate_joints(src_pose=joints[i], dst_pose=joints[j], num_poses=len(jittering))
        joints[i+1:j] = interp_joints
    return joints
            
def linear_interpolate_joints(src_pose, dst_pose, num_poses):
    """
    :param src_pose: [num_joints, 3]
    :param dst_pose: [num_joints, 3]
    :param num_poses: integer
    """
    out_poese = np.zeros((num_poses, src_pose.shape[0], 3))
    for i in range(src_pose.shape[0]):
        zs = []
        for j in range(3):
            x = np.array([0., 1.])
            y = np.array([src_pose[i, j], dst_pose[i, j]])
            f = interpolate.interp1d(x, y, kind="linear")
            z = np.array([float(f(t)) for t in np.arange(0., 1., 1/(num_poses), dtype=np.float)])
            out_poese[:, i, j] = z
    return out_poese   

def interpolate_(poses, target_length):
    """Interpolate the smpl poses. For transl, it conducts lerp, for rotvecs, it conducts slerp.
    :param joints: [seq_len, dim]
    :param target_length: 
    """
    seq_len = poses.shape[0]
    tranl = poses[:, :3]    # [seq_len, 3]
    rotvec = poses[:, 3:]   # [seq_len, dim-3]
    num_rotvec = rotvec.shape[1] // 3
    for i in range(num_rotvec):
        rotvec_in = rotvec[:, i*3:(i+1)*3]
        key_rots = R.from_rotvec(rotvec_in)
        key_times = np.arange(0, target_length, target_length / seq_len, dtype=np.float()).tolist()
        slerp = Slerp(key_times, key_rots)
        times = np.arange(0, target_length, 1, dtype=np.int).tolist()
        interp_rots = slerp(times)
        rotvec_out = interp_rots.as_rotvec()
        
    
    

if __name__ == "__main__":
    data = np.load("joints.npy")
    data = data[0]
    smoothed_data = smooth_joints(data, filter_dim="1d")

