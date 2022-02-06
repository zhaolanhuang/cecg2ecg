import numpy as np
import os
import random
from keras.utils import np_utils
import math
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def evaluate_with_time(annotation_samples, peak_indices):
    ref =  annotation_samples
    test = peak_indices

    TP = 0
    FP = 0
    FN = 0
    i = 0
    j = 0

    diff = list()

    while (i < ref.shape[0] - 1 and j < test.shape[0] - 1):
        t = test[j]
        T = ref[i]
        if (j != test.shape[0] - 1 and i != ref.shape[0] - 1):
            t_prime = test[j + 1]
            T_prime = ref[i + 1]

            if (t < T):
                if (T - t < (150 /4) and (T - t < abs(T - t_prime) or abs(T_prime - t_prime) < abs(T - t_prime))):
                    # match a and A
                    TP = TP + 1
                    # get next t
                    j = j + 1
                    # get next T
                    i = i + 1

                    diff.append(T-t)
                else:
                    # no match for t
                    FP = FP + 1
                    # get next t
                    j = j + 1
            else:
                if (t - T <= (150/4) and (t - T < abs(t - T_prime) or abs(t_prime - T_prime) < abs(t - T_prime))):
                    # match a and A
                    TP = TP + 1
                    # get next t
                    j = j + 1
                    # get next T
                    i = i + 1

                    diff.append(T-t)

                else:
                    # no match for T
                    FN = FN + 1
                    # get next T
                    i = i + 1

    FN = annotation_samples.shape[0] - TP

    diff_sum = np.sum(diff)
    squared_diff_sum = 0
    abs_diff_sum = 0
    for i in diff:
        abs_diff_sum = abs_diff_sum + math.sqrt(i * i)
        squared_diff_sum = squared_diff_sum + (i*i)
    #squared_diff_sum = np.sum(math.sqrt(diff * diff))
    nr_diff = len(diff)

    return TP,FP,FN,diff_sum, abs_diff_sum, squared_diff_sum, nr_diff

def concatenate_training_set_combi_UnoVis(training_set_cECG1, training_set_cECG1_neg, training_set_cECG2, training_set_cECG2_neg,training_set_cECG3, training_set_cECG3_neg):
    # training_set_ECG_comp = training_set_ECG_comp[:,0:iterator]
    # training_set_ECG_neg_comp = training_set_ECG_neg_comp[:,0:iterator]
    # training_set_BP_comp = training_set_BP_comp[:,0:iterator]
    # training_set_BP_neg_comp = training_set_BP_neg_comp[:,0:iterator]

    x_train_1_cECG1 = training_set_cECG1
    y_train_1_cECG1 = np.ones(x_train_1_cECG1.shape[1])
    x_train_0_cECG1 = training_set_cECG1_neg
    y_train_0_cECG1 = np.zeros(x_train_0_cECG1.shape[1])

    x_train_1_cECG2 = training_set_cECG2
    y_train_1_cECG2 = np.ones(x_train_1_cECG2.shape[1])
    x_train_0_cECG2 = training_set_cECG2_neg
    y_train_0_cECG2 = np.zeros(x_train_0_cECG2.shape[1])

    x_train_1_cECG3 = training_set_cECG3
    y_train_1_cECG3 = np.ones(x_train_1_cECG3.shape[1])
    x_train_0_cECG3 = training_set_cECG3_neg
    y_train_0_cECG3 = np.zeros(x_train_0_cECG3.shape[1])

    # concatenate both training set (0,1) to one
    x_train_1 = np.concatenate((x_train_1_cECG1, x_train_1_cECG2, x_train_1_cECG3), axis=0)
    x_train_0 = np.concatenate((x_train_0_cECG1, x_train_0_cECG2, x_train_0_cECG3), axis=0)
    x_train = np.concatenate((x_train_1, x_train_0), axis=1)

    y_train = np.concatenate((y_train_1_cECG1, y_train_0_cECG1))
    y_train = np_utils.to_categorical(y_train, 2)

    # preprocess data
    x_train = x_train.astype('float32')
    x_train = np.swapaxes(x_train, 0, 1)

    # generate test set
    x_test = x_train[(int(x_train.shape[0] / 2)) - 700:(int(x_train.shape[0] / 2)) + 700:1, :]
    y_test = y_train[int((y_train.shape[0] / 2)) - 700:int((y_train.shape[0] / 2)) + 700:1, :]

    return x_train, y_train, x_test, y_test

def generate_training_set_UnoVis_3cECG(directory, directory_str ):
    random.seed(42) # for reproducability
    file_list = np.zeros(200)
    #training sets containing the whole heart-beat/non heart-beat sequences set
    training_set_cECG1_comp = np.zeros((101, 200000))
    training_set_cECG2_comp = np.zeros((101, 200000))
    training_set_cECG3_comp = np.zeros((101, 200000))
    training_set_cECG1_neg_comp = np.zeros((101, 200000))
    training_set_cECG2_neg_comp = np.zeros((101, 200000))
    training_set_cECG3_neg_comp = np.zeros((101, 200000))
    #it lists the start point of a certain file in the training set
    it = dict()
    #nr_ann lists the # of annotations of a certain file
    nr_ann = dict()
    iterator = 0
    x = 0
    temp_sig1 = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        
        if len(filename) < 11:
            continue

        if (filename[0] != 'a' and filename[0:5] != 'cECG2' and filename[0:5] != 'cECG3' and filename[10] != 'f' and filename.startswith('cECG1') ):
            file_list[x] = int(filename[21:-8])
#             print(filename)
            x = x + 1

            # open signal and annotation files
            a = np.load(directory_str + 'cECG1' + filename[5:])
            b = np.load(directory_str + 'cECG2' + filename[5:])
            c = np.load(directory_str + 'cECG3' + filename[5:])
            sig = np.zeros((a.shape[0],3))
            sig[:,0] = a
            sig[:,1] = b
            sig[:,2] = c
            annotation_array = np.load(directory_str + 'annotations_' + filename[5:])

            waves_cECG1 = np.zeros((4, sig.shape[0]))
            waves_cECG2 = np.zeros((4, sig.shape[0]))
            waves_cECG3 = np.zeros((4, sig.shape[0]))


            waves_cECG1[2, :] = butter_bandpass_filter(sig[:, 0], 8, 20, 250, order=2)
            waves_cECG2[2, :] = butter_bandpass_filter(sig[:, 1], 8, 20, 250, order=2)
            waves_cECG3[2, :] = butter_bandpass_filter(sig[:, 2], 8, 20, 250, order=2)
            # Bug Fixed 0826
            annotation_array = annotation_array.reshape(-1)
#             print(annotation_array.shape)

            for i in range(0, annotation_array.shape[0] - 1, 1):
                if i > 0 and i < annotation_array.shape[0] and annotation_array[i] - annotation_array[i - 1] > 140 and \
                                        annotation_array[i] - annotation_array[i + 1] < -140:
                    # generation of a random false example
                    a = [random.randint(20, annotation_array[i] - annotation_array[i - 1] - 120),
                         random.randint(annotation_array[i] - annotation_array[i + 1] + 120, -20)]
                    b = random.randint(0, 1)
                    neg_interv = a[b]
                else:
                    neg_interv = 30

                if waves_cECG1[2, annotation_array[i] - 50:annotation_array[i] + 51:1].shape[0] == 101:
                    training_set_cECG1_comp[:, iterator + i] = waves_cECG1[2,
                                                             annotation_array[i] - 50:annotation_array[i] + 51:1]
                    training_set_cECG2_comp[:, iterator + i] = waves_cECG2[2,
                                                            annotation_array[i] - 50:annotation_array[i] + 51:1]
                    training_set_cECG3_comp[:, iterator + i] = waves_cECG3[2,
                                                            annotation_array[i] - 50:annotation_array[i] + 51:1]
                else:
                    continue

                if waves_cECG1[2, annotation_array[i] - 50 - neg_interv:annotation_array[i] + 51 - neg_interv:1].shape[0] == 101:
                    training_set_cECG1_neg_comp[:, iterator + i] = waves_cECG1[2,
                                                                 annotation_array[i] - 50 - neg_interv:annotation_array[
                                                                                                           i] + 51 - neg_interv:1]
                    training_set_cECG2_neg_comp[:, iterator + i] = waves_cECG2[2,
                                                                annotation_array[i] - 50 - neg_interv:annotation_array[
                                                                                                          i] + 51 - neg_interv:1]
                    training_set_cECG3_neg_comp[:, iterator + i] = waves_cECG3[2,
                                                            annotation_array[i] - 50 - neg_interv:annotation_array[i] + 51  - neg_interv:1]
                else:
                    continue

            it[int(filename[21:-8])] = iterator
            nr_ann[int(filename[21:-8])] = annotation_array.shape[0]
            iterator = iterator + annotation_array.shape[0]
            continue
        else:
            continue

    training_set_cECG1_comp = training_set_cECG1_comp[:,0:iterator]
    training_set_cECG2_comp = training_set_cECG2_comp[:,0:iterator]
    training_set_cECG3_comp = training_set_cECG3_comp[:,0:iterator]
    training_set_cECG1_neg_comp = training_set_cECG1_neg_comp[:,0:iterator]
    training_set_cECG2_neg_comp = training_set_cECG2_neg_comp[:,0:iterator]
    training_set_cECG3_neg_comp = training_set_cECG3_neg_comp[:,0:iterator]

    return training_set_cECG1_comp, training_set_cECG2_comp, training_set_cECG3_comp, training_set_cECG1_neg_comp, training_set_cECG2_neg_comp, training_set_cECG3_neg_comp, it, iterator, nr_ann, file_list
