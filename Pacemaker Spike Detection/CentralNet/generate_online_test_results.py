import numpy as np
import scipy.io
import peakutils
import time
import sys
sys.path.append('..')
from common import data_loader, utils

from sacred import Experiment
from sacred.observers import MongoObserver, TelegramObserver


now_str = time.strftime('%Y-%m-%d-%H_%M_%S',time.localtime(time.time()))
ex = Experiment(f'OnlineTest-GenerateResult')
ex.observers.append(MongoObserver(db_name='Pacemaker'))
telegram_obs = TelegramObserver.from_config('../common/telegram_config.json')
ex.observers.append(telegram_obs)


@ex.config
def common_config():
    unovis_path = f'../../Dataset/UnoViS_pacemaker2020.mat'
    model_dir = f'./Model_diff_2021-10-24-10_58_16'
    isNormalization = False
    maxfilt_w = 0 # Window size of maximum filter, 0 means not enabled
    exp_time = now_str
    input_type = 'diff' # diff - only using differential cecg; original - only using cecg; hybrid - using both


@ex.capture
def load_data_and_normalization(unovis_path, rec_no, input_type, isNormalization):
    """
    Load the record and annotations of spikes from UnoVis Pacemaker2020
    """
    w = 129
    rec_no = rec_no - 1
    mat = scipy.io.loadmat(unovis_path)
    cecg1 = mat['unovis'][0, rec_no]['channels'][0, 1]['data'][:, 0]
    cecg2 = mat['unovis'][0, rec_no]['channels'][0, 2]['data'][:, 0]
    cecg3 = mat['unovis'][0, rec_no]['channels'][0, 3]['data'][:, 0]

    cecg1_diff = utils.differentiator(cecg1)
    cecg2_diff = utils.differentiator(cecg2)
    cecg3_diff = utils.differentiator(cecg3)
    if input_type == 'original':
        cecg_all = np.array([cecg1, cecg2, cecg3])
    elif input_type == 'diff':
        cecg_all = np.array([cecg1_diff, cecg2_diff, cecg3_diff])
    else:
        cecg_all = np.array([cecg1, cecg2, cecg3, cecg1_diff, cecg2_diff, cecg3_diff])

    from tensorflow.keras.preprocessing import timeseries_dataset_from_array
    dataset = timeseries_dataset_from_array(
            cecg_all.T, None, w, batch_size=1)

    x = np.array(list(dataset.as_numpy_iterator()))
    x = np.squeeze(x, axis=1)
#     x = cecg_all.swapaxes(1,3)

    ann = mat['unovis'][0, rec_no]['channels'][0, 0]['ann']
    ty = ann['type']
    valid_idx = np.where(ty[0, :] != 'TA') # TA is noise annotation, not pacemaker spikes
    y = ann['loc'].astype(float)[0]
    locs = y[valid_idx]

    return x, locs


@ex.automain
def define_and_test(_run, model_dir, input_type, maxfilt_w):

    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Input, concatenate, Activation
    from tensorflow.keras.layers import Conv1D, MaxPool1D, LSTM, InputLayer, GRU, BatchNormalization, GlobalAveragePooling1D
    from tensorflow.keras.models import Model, load_model, Sequential
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard, Callback
    from tensorflow.keras.optimizers import Adam, Adadelta
    from layers import WeightedSum, ApplyUniformLayer
    import pandas as pd

    result_dir = f'./OnlineTest_{input_type}_{now_str}'
    import os
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    stat_dict = {'Subject':[], 'Rec':[],'Th':[],'TP':[], 'FP':[], 'FN':[],'squared_diff_sum':[]}
    for patient_for_test in range(1,21):
        print(f'Patient:{patient_for_test}')
        model_path = f'{model_dir}/model_fusionNet_test_patient{patient_for_test}.h5'
        model = load_model(model_path, custom_objects={'WeightedSum': WeightedSum})
        recs = data_loader.patient2rec[patient_for_test]
        for rec_no in recs:
            print(f'Rec:{rec_no}')
            x, ref_locs = load_data_and_normalization(rec_no=rec_no)
            print(x.shape)
            y = model.predict(x)
            
            y_fus = y[0][:,1]
            y_pred = np.pad(y_fus, 64, constant_values=0)
            
            import scipy.ndimage as ndimage
            if maxfilt_w > 0:
                y_pred_maxfilt = ndimage.maximum_filter1d(y_pred, maxfilt_w)
                y_pred = y_pred_maxfilt

            th2locs = dict()
            th2locs['y_pred'] = y_pred
            th2locs['ref_locs'] = ref_locs
            for t in np.arange(0.1, 1, 0.01):
                indices = peakutils.indexes(y_pred, thres=t, min_dist=20)
                th2locs[f'Idx{round(t*100)}'] = indices
                TP, FP, FN, diff_sum, abs_diff_sum, squared_diff_sum, nr_diff = utils.evaluate_with_time(
                    ref_locs, indices)

                stat_dict['Subject'].append(patient_for_test)
                stat_dict['Rec'].append(rec_no)
                stat_dict['Th'].append(t)
                stat_dict['TP'].append(TP)
                stat_dict['FP'].append(FP)
                stat_dict['FN'].append(FN)
                stat_dict['squared_diff_sum'].append(squared_diff_sum)
            scipy.io.savemat(f'{result_dir}/detected_spikes_rec{rec_no}.mat', th2locs)
    stat_res = pd.DataFrame(stat_dict)
    stat_res.to_csv(f'{result_dir}/statistic.csv')

    return None