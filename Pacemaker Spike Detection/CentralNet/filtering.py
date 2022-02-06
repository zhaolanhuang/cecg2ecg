import scipy.io, scipy.signal
import numpy as np

def supress_motion_artifact(signal, fs, median_w=0.150):
    """
    Use the median filter for supress the motion artifacts
    
    signal: array-like (n_timestep, n_features)
    fs : int, sample rate of inputs
    median_w : float, window length of median filter in second
    """
    _signal = signal.copy()
    med_kel_size = round(median_w * fs)
    med_kel_size = med_kel_size + 1 if (med_kel_size % 2) == 0 else med_kel_size
    _signal = _signal - scipy.signal.medfilt(_signal, [med_kel_size,1])
    return _signal

def desaturate(signal, fs, lower_limit, upper_limit, reflation=0.1):
    """
    Eliminate the saturated part in signal.
    
    signal : array-like (n_timestep, n_features)
    fs : int, sample rate of inputs
    upper_limit, lower_limit : float, upper and lower limit of signal
    reflation : float, reflation in second
    
    """
    _signal = signal.copy()
    rf_pts = round(reflation * fs)
    up_idx = np.argwhere(signal>upper_limit)
    lw_idx = np.argwhere(signal<lower_limit)
    
    def _func(i):
        idx = np.arange(i[0] - rf_pts, i[0] + rf_pts)
        idx = np.vstack((idx,idx)).reshape(-1,2)
        idx[:,1] = i[1]
        return idx
    if up_idx.shape[0] > 0 :
        up_idx = np.apply_along_axis(_func, 1, up_idx).reshape(-1,2)
    if lw_idx.shape[0] > 0 :
        lw_idx = np.apply_along_axis(_func, 1, lw_idx).reshape(-1,2)

    sat_idx = np.concatenate((up_idx, lw_idx)).reshape(-1,2)
    sat_idx = np.unique(sat_idx,axis=0)
    sat_idx = sat_idx[(sat_idx[:,0] >= 0) & (sat_idx[:,0] < signal.shape[0])]
    
    _signal[sat_idx] = 0
    return _signal, sat_idx