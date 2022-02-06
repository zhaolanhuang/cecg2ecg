import numpy as np
import scipy.io, scipy.signal
from sklearn.preprocessing import MinMaxScaler,StandardScaler

patient2rec = dict() # patient id to recording id
patient2rec[1] = [1,2,3,6,24]
patient2rec[2] = [4,7,8,17,21]
patient2rec[3] = [5,9,10,25]
patient2rec[4] = [11,12,13,14,15,26,27,28,29,30,31]
patient2rec[5] = [16,18,19,20]
patient2rec[6] = [22,23]

def load_auto2012(rec_no, mat_path):
    """
    Load Record from dataset UnoVis Auto2012
    
    rec_no : int, record number, range 1 to 31
    mat_path : str, path to mat-file of dataset
    
    Return
    cecg : ndarray (timestep, cecg_channel), 3-channel cecg
    recg : ndarray (timestep, 1), 1-channel rECG
    fs : int, sample rate
    r_loc : ndarray (#R-peaks,), manually marked position of R-peaks
    """
    unovis = scipy.io.loadmat(mat_path)
    rec_idx = rec_no - 1
    fs = unovis['unovis'][0][rec_idx]['channels']['fs'][0,0][0,0]
    cecg1 = unovis['unovis'][0][rec_idx]['channels']['data'][0,:][0]
    cecg2 = unovis['unovis'][0][rec_idx]['channels']['data'][0,:][1]
    cecg3 = unovis['unovis'][0][rec_idx]['channels']['data'][0,:][2]
    recg = unovis['unovis'][0][rec_idx]['channels']['data'][0,:][3]
    cecg = np.hstack([cecg1, cecg2, cecg3])
    r_loc = unovis['unovis'][0][rec_idx]['channels']['ann'][0,:][3]['loc'][0,1][:,0]
    
    return cecg, recg, fs, r_loc

def load_qrs_scaler(test_patient, cycles_data_path, scaler_class=StandardScaler):
    training_rec = set(range(1,32)) - set(patient2rec[test_patient])
    cyc_data = scipy.io.loadmat(cycles_data_path)
    cycles = cyc_data['extracted_cycles'][0]
    training_cycles = []
    for i in training_rec: 
        idx = i - 1
        for j in range(0, cycles[idx]['cycles']['data'].shape[1]):
            training_cycles.append(cycles[idx]['cycles']['data'][0,j])
    training_cycles = np.hstack(training_cycles).swapaxes(0,1)
    scaler_cecg = scaler_class()
    scaler_cecg.fit(training_cycles[:, 0:3])
    return scaler_cecg

def load_lstm_scaler(test_patient, data_path, scaler_class=StandardScaler):
    mat = scipy.io.loadmat(data_path)
    recordings = mat['recordings'][0]
    training_rec = set(range(1,32)) - set(patient2rec[test_patient])
    train = []
    for i in training_rec:
        idx = i - 1
        numOfseg = recordings[idx][1][0].shape[0]
        for j in range(0, numOfseg):
            train.append(np.expand_dims(recordings[idx][1][0][j][:,:], axis=0))
    train = np.vstack(train)
    train = train.swapaxes(1,2)
    x_train = train[:,:,0:3]
    _x = x_train.reshape(-1,3)
    scaler_cecg = scaler_class()
    scaler_cecg.fit(_x)
    return scaler_cecg
    