import numpy as np
import scipy.io, scipy.signal
from tensorflow.keras.utils import to_categorical


def load_data_for_lstm(mat_path, patient_for_test, isNormalization):
    x_train, y_train, x_test, y_test = None, None, None, None
    return x_train, y_train, x_test, y_test


def load_data_for_cnn(mat_path, patient_for_test, isNormalization):
    """
    Load Data from MAT-file for training, generate Training and test data according
    to leave-one-out regulation.
    Argument
    patient_for_test: Subject ID for test, the corresponding records are not used for training

    Return
    x_train, y_train: training data. x_train = [cecg1, cecg2, cecg3, cecg1_diff, cecg2_diff, cecg3_diff]
    x_test, y_test: test data. x_train = [cecg1, cecg2, cecg3, cecg1_diff, cecg2_diff, cecg3_diff]
    """
    x_train, y_train, x_test, y_test = None, None, None, None
    mat = scipy.io.loadmat(mat_path)
    Fs = mat['Fs'][0][0]
    recordings = mat['recordings'][0]
    training_rec = set(range(1,98)) - set(patient2rec[patient_for_test])
    x_train = []
    y_train = []
    for i in training_rec:
        idx = i - 1
        numOfsample = recordings[idx]['positive'][0].shape[0]
        y_train.append(np.ones(numOfsample))
        for j in range(0, numOfsample):
            x_train.append(np.expand_dims(recordings[idx]['positive'][0][j], axis=0))

        numOfsample = recordings[idx]['negative'][0].shape[0]
        y_train.append(np.zeros(numOfsample))
        for j in range(0, numOfsample):
            x_train.append(np.expand_dims(recordings[idx]['negative'][0][j], axis=0))
    x_train = np.vstack(x_train).swapaxes(1,2)
    y_train = np.hstack(y_train)
    y_train = to_categorical(y_train,2)

    # Shuffle the training data
    rng = np.random.default_rng(42) # for reproducibility
    shuffle_idx = np.arange(0, x_train.shape[0])
    rng.shuffle(shuffle_idx)
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    x_test = []
    y_test = []
    for i in patient2rec[patient_for_test]:
        idx = i - 1
        numOfsample = recordings[idx]['positive'][0].shape[0]
        y_test.append(np.ones(numOfsample))
        for j in range(0, numOfsample):
            x_test.append(np.expand_dims(recordings[idx]['positive'][0][j], axis=0))

        numOfsample = recordings[idx]['negative'][0].shape[0]
        y_test.append(np.zeros(numOfsample))
        for j in range(0, numOfsample):
            x_test.append(np.expand_dims(recordings[idx]['negative'][0][j], axis=0))
    x_test = np.vstack(x_test).swapaxes(1,2)
    y_test = np.hstack(y_test)
    y_test = to_categorical(y_test,2)
   
    return x_train, y_train, x_test, y_test

# Mapping Subject ID to record ID
patient2rec = {  1 : [1,2,3,4,5,6,7,8,9],
            2 : [10,11,12,13,],
            3 : [14,15,],
            4 : [16,17,18,19,20,21,22,23,24],
            5 : [25,26,27,28,],
            6 : [29,30,31,32,33],
            7 : [34,35,36,37,38],
            8 : [40,41,39],
            9 : [42,43,44,45,46,47,],
            10 : [49,48,50,51,52],
            11 : [53,54,55,56,57,58,59],
            12 : [60,61,62,63,64,65,],
            13 : [66,67,68],
            14 : [69,70,71],
            15 : [72,73,],
            16 : [74,75,76,77,78,79,80,81,82],
            17 : [83,84,85,86,87],
            18 : [88,89,],
            19 : [90,91,92,93,],
            20 : [94,95,96,97],
            'ONLY_ERR_SPKS_REC' : [39,49,55,62,86,94,]}