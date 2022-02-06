import os
import numpy as np
import peakutils
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten, Input, concatenate
from tensorflow.keras.models import Model, load_model
import gc
from tensorflow.keras import backend as K
import logging as log

from unovis.utils import numpy_table_to_beautiful_string
import unovis.help_functions as help_functions

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

    log.info("Generating training data.")
    [training_set_cECG1_comp, training_set_cECG2_comp, training_set_cECG3_comp, training_set_cECG1_neg_comp, training_set_cECG2_neg_comp, training_set_cECG3_neg_comp, it, iterator, nr_ann, file_list] = help_functions.generate_training_set_UnoVis_3cECG(directory, signal_directory)

    file_list = file_list[0:31]
    file_list = [int(f) for f in file_list]

    TP = np.empty((len(file_list), len(epochs), len(thresh)), dtype=object)
    FP = np.empty((len(file_list), len(epochs), len(thresh)), dtype=object)
    FN = np.empty((len(file_list), len(epochs), len(thresh)), dtype=object)
    detected_indices = np.empty((len(file_list), len(epochs), len(thresh)), dtype=object)
    diff_sum = np.empty((len(file_list), len(epochs), len(thresh)), dtype=object)
    abs_diff_sum = np.empty((len(file_list), len(epochs), len(thresh)), dtype=object)
    squared_diff_sum = np.empty((len(file_list), len(epochs), len(thresh)), dtype=object)
    nr_diff = np.empty((len(file_list), len(epochs), len(thresh)), dtype=object)

    for f_idx, f in enumerate(file_list):
        test_file = 'UnoViS_auto2012_' + str(int(f)) + '.hea.npy'

        training_set_cECG1 = np.zeros((101, 1))
        training_set_cECG1_neg = np.zeros((101, 1))
        training_set_cECG2 = np.zeros((101, 1))
        training_set_cECG2_neg = np.zeros((101, 1))
        training_set_cECG3 = np.zeros((101, 1))
        training_set_cECG3_neg = np.zeros((101, 1))
        for i in file_list:
            if subject_id[i] != subject_id[f]:
                training_set_cECG1 = np.concatenate((training_set_cECG1, training_set_cECG1_comp[:, it[i]:it[i] + nr_ann[i]]), axis=1)
                training_set_cECG1_neg = np.concatenate((training_set_cECG1_neg, training_set_cECG1_neg_comp[:, it[i]:it[i] + nr_ann[i]]), axis=1)
                training_set_cECG2 = np.concatenate((training_set_cECG2, training_set_cECG2_comp[:, it[i]:it[i] + nr_ann[i]]), axis=1)
                training_set_cECG2_neg = np.concatenate((training_set_cECG2_neg, training_set_cECG2_neg_comp[:, it[i]:it[i] + nr_ann[i]]), axis=1)
                training_set_cECG3 = np.concatenate((training_set_cECG3, training_set_cECG3_comp[:, it[i]:it[i] + nr_ann[i]]), axis=1)
                training_set_cECG3_neg = np.concatenate((training_set_cECG3_neg, training_set_cECG3_neg_comp[:, it[i]:it[i] + nr_ann[i]]), axis=1)

        x_train, y_train, x_test, y_test = help_functions.concatenate_training_set_combi_UnoVis(training_set_cECG1, training_set_cECG1_neg, training_set_cECG2, training_set_cECG2_neg, training_set_cECG3, training_set_cECG3_neg)

        x_train = np.expand_dims(x_train, axis=2)
        x_test = np.expand_dims(x_test, axis=2)

        for ep_idx, ep in enumerate(epochs):
            gc.collect()
            try:
                model = load_model(signal_directory + 'model_s' + str(int(subject_id[f])) + '_e' + str(int(ep)) + '.h5')
            except:
                cECG1_inputs = Input(shape=(101, 1))
                x = Conv1D(10, 20, activation='relu')(cECG1_inputs)
                x = MaxPool1D(pool_size=2)(x)
                x = Conv1D(20, 15, activation='relu', padding='same')(x)
                x = Conv1D(25, 10, activation='relu', padding='same')(x)
                x = Conv1D(30, 10, activation='relu', padding='same')(x)
                x = Flatten()(x)

                cECG2_inputs = Input(shape=(101, 1))
                y = Conv1D(10, 20, activation='relu')(cECG2_inputs)
                y = MaxPool1D(pool_size=2)(y)
                y = Conv1D(20, 15, activation='relu', padding='same')(y)
                y = Conv1D(25, 10, activation='relu', padding='same')(y)
                y = Conv1D(30, 10, activation='relu', padding='same')(y)
                y = Flatten()(y)

                cECG3_inputs = Input(shape=(101, 1))
                z = Conv1D(10, 20, activation='relu')(cECG3_inputs)
                z = MaxPool1D(pool_size=2)(z)
                z = Conv1D(20, 15, activation='relu', padding='same')(z)
                z = Conv1D(25, 10, activation='relu', padding='same')(z)
                z = Conv1D(30, 10, activation='relu', padding='same')(z)
                z = Flatten()(z)

                combi = concatenate([x, y, z])
                combi = Dense(100, activation='relu')(combi)
                combi = Dropout(0.2)(combi)
                combi = Dense(40, activation='relu')(combi)
                combi = Dropout(0.2)(combi)
                output = Dense(2, activation='softmax')(combi)

                model = Model(inputs=[cECG1_inputs, cECG2_inputs, cECG3_inputs], outputs=[output])

                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])
                # fit model to training data
                c1 = x_train[:, 0:101]
                c2 = x_train[:, 101:202]
                c3 = x_train[:, 202:]


                model.fit([np.concatenate((c1, c2, c3), axis=0),
                           np.concatenate((c2, c3, c1), axis=0),
                           np.concatenate((c3, c1, c2), axis=0)],
                          np.concatenate((y_train, y_train, y_train), axis=0),
                          batch_size=512, epochs=ep, verbose=1)
                model.save(signal_directory + 'model_s' + str(int(subject_id[f])) + '_e' + str(int(ep)) + '.h5')

            # apply neural network as peak detection algorithm
            log.info("Building evaluation data:")
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
                detected_indices[f_idx, ep_idx, th_idx] = indices

                # evaluation
                TP[f_idx, ep_idx, th_idx], FP[f_idx, ep_idx, th_idx], FN[f_idx, ep_idx, th_idx], diff_sum[f_idx, ep_idx, th_idx], abs_diff_sum[f_idx, ep_idx, th_idx], squared_diff_sum[f_idx, ep_idx, th_idx], nr_diff[f_idx, ep_idx, th_idx] = help_functions.evaluate_with_time(annotation_array, indices)

            K.clear_session()

    return TP, FP, FN, nr_diff, diff_sum, abs_diff_sum, squared_diff_sum

def save_results(signal_directory, epochs, thresh, TP, FP, FN, nr_diff, diff_sum, abs_diff_sum, squared_diff_sum):
    np.save(signal_directory + 'epochs.npy', epochs)
    np.save(signal_directory + 'thresh.npy', thresh)
    np.save(signal_directory + 'TP.npy', TP)
    np.save(signal_directory + 'FP.npy', FP)
    np.save(signal_directory + 'FN.npy', FN)
    np.save(signal_directory +'nr_diff.npy', nr_diff)
    np.save(signal_directory + 'diff_sum.npy', diff_sum)
    np.save(signal_directory +'abs_diff_sum.npy', abs_diff_sum)
    np.save(signal_directory + 'squared_diff_sum.npy', squared_diff_sum)   

def load_results(signal_directory):
    epochs = np.load(signal_directory + 'epochs.npy', allow_pickle=True)
    thresh = np.load(signal_directory + 'thresh.npy', allow_pickle=True)
    TP = np.load(signal_directory + 'TP.npy', allow_pickle=True)
    FP = np.load(signal_directory + 'FP.npy', allow_pickle=True)
    FN = np.load(signal_directory + 'FN.npy', allow_pickle=True)
    nr_diff = np.load(signal_directory + 'nr_diff.npy', allow_pickle=True)
    diff_sum = np.load(signal_directory + 'diff_sum.npy', allow_pickle=True)
    abs_diff_sum = np.load(signal_directory + 'abs_diff_sum.npy', allow_pickle=True)
    squared_diff_sum = np.load(signal_directory + 'squared_diff_sum.npy', allow_pickle=True)   

    return epochs, thresh, TP, FP, FN, nr_diff, diff_sum, abs_diff_sum, squared_diff_sum

def evaluation(epochs, thresh, TP, FP, FN, nr_diff, diff_sum, abs_diff_sum, squared_diff_sum):
    TP = np.sum(TP, axis=0, dtype=np.float64)
    FP = np.sum(FP, axis=0, dtype=np.float64)
    FN = np.sum(FN, axis=0, dtype=np.float64)

    nr_diff = np.sum(nr_diff, axis=0, dtype=np.float64)
    diff_sum = np.divide(np.sum(diff_sum, axis=0), nr_diff + 1e-9)
    abs_diff_sum = np.divide(np.sum(abs_diff_sum, axis=0), nr_diff+ 1e-9)
    squared_diff = np.sum(squared_diff_sum, axis=0, dtype=np.float64)
    rms_diff = np.sqrt(np.divide(squared_diff, nr_diff+ 1e-9, dtype=np.float64))

    sens, pred, final = calc_measures(TP, FP, FN)
    print_results(epochs, thresh, TP, FP, FN, diff_sum, abs_diff_sum, rms_diff, sens, pred, final)

def print_results(epoch, thresh, TP, FP, FN, diff_sum, abs_diff_sum, rms_diff, sens, pred, final, fs=250):
    print(numpy_table_to_beautiful_string(TP, epoch, thresh, "EP", "TH", header="TP", col_width=8))
    print()
    print(numpy_table_to_beautiful_string(FP, epoch, thresh, "EP", "TH", header="FP", col_width=8))
    print()
    print(numpy_table_to_beautiful_string(FN, epoch, thresh, "EP", "TH", header="FN", col_width=8))
    print()
    print(numpy_table_to_beautiful_string(diff_sum * 1000/fs, epoch, thresh, "EP", "TH", col_width=6, header="diff_sum [ms]"))
    print()
    print(numpy_table_to_beautiful_string(abs_diff_sum * 1000/fs, epoch, thresh, "EP", "TH", header="abs_diff_sum [ms]"))
    print()
    print(numpy_table_to_beautiful_string(rms_diff * 1000/fs, epoch, thresh, "EP", "TH", header="rms_diff [ms]"))
    print()
    print(numpy_table_to_beautiful_string(sens, epoch, thresh, "EP", "TH", header="Sensitivity [%]"))
    print()
    print(numpy_table_to_beautiful_string(pred, epoch, thresh, "EP", "TH", header="Predictivity [%]"))
    print()
    print(numpy_table_to_beautiful_string(final, epoch, thresh, "EP", "TH", header="Final Score [%]"))
    print()

def calc_measures(TP, FP, FN):
    sensitivity = 100*np.divide(TP, np.add(TP, FN))
    predictivity = 100*np.divide(TP, np.add(TP, FP))
    final_score = 0.5*np.add(sensitivity, predictivity)

    return sensitivity, predictivity, final_score
