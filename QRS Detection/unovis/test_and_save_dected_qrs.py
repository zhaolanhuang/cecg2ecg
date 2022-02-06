import os
import numpy as np
import peakutils
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten, Input, concatenate
from tensorflow.keras.models import Model, load_model
import gc
from tensorflow.keras import backend as K
import logging as log

from utils import numpy_table_to_beautiful_string
import help_functions as help_functions

subject_id = dict()
subject_id[1] = 1
subject_id[2] = 1
subject_id[3] = 1
subject_id[4] = 2
subject_id[5] = 3
subject_id[6] = 1
subject_id[7] = 2
subject_id[8] = 2
subject_id[9] = 3
subject_id[10] = 3
subject_id[11] = 4
subject_id[12] = 4
subject_id[13] = 4
subject_id[14] = 4
subject_id[15] = 4
subject_id[16] = 5
subject_id[17] = 2
subject_id[18] = 5
subject_id[19] = 5
subject_id[20] = 5
subject_id[21] = 2
subject_id[22] = 6
subject_id[23] = 6
subject_id[24] = 1
subject_id[25] = 3
subject_id[26] = 4
subject_id[27] = 4
subject_id[28] = 4
subject_id[29] = 4
subject_id[30] = 4
subject_id[31] = 4

def train_UnoVis_3cECG(signal_directory, epochs, thresh):
    directory = os.fsencode(signal_directory)
    epochs = np.array(epochs)
    thresh = np.array(thresh)
    i = 0
    it = dict()
    nr_ann = dict()
    x = 0

    for f in range(1, 32):
        test_file = 'UnoViS_auto2012_' + str(int(f)) + '.hea.npy'

        for ep_idx, ep in enumerate(epochs):
            gc.collect()
            model = load_model(signal_directory + 'model_s' + str(int(subject_id[f])) + '_e' + str(int(ep)) + '.h5')
            
            # apply neural network as peak detection algorithm
            log.info(f'Building evaluation data of rec {f}:')
            a = np.load(signal_directory + 'cECG1' + test_file)
            b = np.load(signal_directory + 'cECG2' + test_file)
            c = np.load(signal_directory + 'cECG3' + test_file)
            sig = np.zeros((a.shape[0], 3))
            sig[:, 0] = a
            sig[:, 1] = b
            sig[:, 2] = c

            waves_cECG1 = np.zeros((4, sig.shape[0]))
            waves_cECG2 = np.zeros((4, sig.shape[0]))
            waves_cECG3 = np.zeros((4, sig.shape[0]))

            waves_cECG1[2, :] = help_functions.butter_bandpass_filter(sig[:, 0], 8, 20, 250, order=2)
            waves_cECG2[2, :] = help_functions.butter_bandpass_filter(sig[:, 1], 8, 20, 250, order=2)
            waves_cECG3[2, :] = help_functions.butter_bandpass_filter(sig[:, 2], 8, 20, 250, order=2)

            annotation_array = np.load(signal_directory + 'annotations_' + test_file)
            annotation_array = annotation_array.reshape(-1)

            test_samples=np.zeros((waves_cECG1.shape[1],101*3))

            #prediction of the r-peaks in the whole signal
            for i in range(50, waves_cECG1.shape[1]- 51):
                test_sample_cECG1 = waves_cECG1[2, i - (50):i + (51):1]
                test_sample_cECG2 = waves_cECG2[2, i - (50):i + (51):1]
                test_sample_cECG3 = waves_cECG3[2, i - (50):i + (51):1]
                test_sample_cECG1.resize(1, test_sample_cECG1.shape[0])
                test_sample_cECG2.resize(1, test_sample_cECG2.shape[0])
                test_sample_cECG3.resize(1, test_sample_cECG3.shape[0])
                test_sample = np.concatenate((test_sample_cECG1, test_sample_cECG2, test_sample_cECG3), axis=1)
                test_samples[i,:] = test_sample

            #detect peaks from prediction
            test_samples = np.expand_dims(test_samples, axis=2)

            log.info("Predict on evaluation data:")
            prediction = model.predict([test_samples[:, 0:101 , :], test_samples[:, 101:202 , :], test_samples[:, 202: , :]])

            prediction = prediction.swapaxes(0,1)

            log.info("Thershold detection:")
            for th_idx, th in enumerate(thresh):
                indices = peakutils.indexes(prediction[1,:], thres=th, min_dist=100)
                np.save(f'result/detected_qrs_rec{int(f)}_ep{int(ep)}_th{th}.npy', indices)

            K.clear_session()


if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    train_UnoVis_3cECG(f'../cache/', [8], [0.7])
