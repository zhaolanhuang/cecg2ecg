import numpy as np
from scipy.signal import resample
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model, clone_model

def extract_qrs_complex_with_interpolation(ecg, r_loc, qrs_len, intpl_len):
    """
    Extract QRS Complex and interpolate them to determined length
    ecg : ndarray (timestep, channels) , ECG signal
    r_loc : ndarray (#R peaks, ), Position of R peaks
    qrs_len : ndarray (#R peaks, ), Corresponding QRS length
    intpl_len : int, output length of each interpolated QRS Complex
    """
    qrs_compls = []
    N = ecg.shape[0]
    for _pos,_len in zip(r_loc, qrs_len):
        begin = int(_pos - (_len * 1 / 3.0))
        begin = np.maximum(begin, 0)
        end = int(_pos + (_len * 2 / 3.0))
        end = np.minimum(end, N-1)
        _ecg = resample(ecg[begin:end,:], intpl_len) 
        qrs_compls.append(_ecg)
    qrs_compls = np.array(qrs_compls)
    return qrs_compls
    
    
def extract_qrs_complex_with_padding(ecg, r_loc, qrs_len, pad_len, pad_val=-100):
    """
    Extract QRS Complex and pad them at the end with constant value
    ecg : ndarray (timestep, channels) , ECG signal
    r_loc : ndarray (#R peaks, ), Position of R peaks
    qrs_len : ndarray (#R peaks, ), Corresponding QRS length
    pad_len : int, output length of each padded QRS Complex
    pad_val : float, The values to set the padded values
    """
    qrs_compls = []
    for _pos,_len in zip(r_loc, qrs_len):
        begin = int(_pos - (_len * 1 / 3.0))
        begin = np.maximum(begin, 0)
        end = int(_pos + (_len * 2 / 3.0))
        end = np.minimum(end, N-1)
        _ecg = pad_sequences(ecg[begin:end,:].swapaxes(0,1),
                       padding='post', maxlen=pad_len,value=pad_val,dtype=float).swapaxes(0,1)
        qrs_compls.append(_ecg)
    qrs_compls = np.array(qrs_compls)
    return qrs_compls

def calculate_parameter_switch_point(r_loc, qrs_len):
    para_switch_point = []
    for _pos,_len in zip(r_loc, qrs_len):
        para_switch_point.append(np.maximum(int(_pos - (_len * 1 / 3.0)),0))
    return np.array(para_switch_point)

def resample_R_location(r_loc, fs, resample_fs):
    resample_factor = resample_fs / fs
    return np.round(r_loc * resample_factor)

def load_model_with_stateful(model_path, batch_size=1):
    model = load_model(model_path)
    for layer in model.layers:
        if hasattr(layer, 'stateful'):
            setattr(layer, 'stateful', True)
    newModel = clone_model(model, 
           input_tensors=Input(shape=(*model.input.shape[1:],), batch_size=batch_size))
    newModel.load_weights(model_path)
    return newModel