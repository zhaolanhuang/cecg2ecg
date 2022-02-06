import numpy as np
import scipy.io
from math import pi
import time
from gauss_layer import GaussLayer
import numpy as np
import utils

from sacred import Experiment
from sacred.observers import MongoObserver, TelegramObserver

now_str = time.strftime('%Y-%m-%d-%H_%M_%S',time.localtime(time.time()))
ex = Experiment(f'Direct-Fusion')
ex.observers.append(MongoObserver())
telegram_obs = TelegramObserver.from_config('telegram_config.json')
ex.observers.append(telegram_obs)



@ex.config
def common_config():
    unovis_path = f'UnoViS_auto2012.mat'
    cycles_data_path = 'extracted_cycles_resampled.mat'
    lstm_data_path = 'dataset_LSTM_seg_10s_with_recording_nr.mat'
    
    lstm_model_dir = f'../Reconstruction/LSTM/Model_100S64N'
    para_est_model_dir = f'../Reconstruction/CNNwReconstructionLoss/Model_'
    
    is_lstm_normalization = True
    amp_correction = False
    patient_for_test = -1
    exp_time = now_str


def ecg_dyn(theta_k, z_k, omega, delta, a, thetai, b):
    """
    ECG dynamic Model from 
    [1]P. E. McSharry, G. D. Clifford, L. Tarassenko, and L. A. Smith, “A dynamical model for generating synthetic electrocardiogram signals,” IEEE Transactions on Biomedical Engineering, vol. 50, no. 3, pp. 289–294, Mar. 2003, doi: 10.1109/TBME.2003.808805.
    """
    theta = (theta_k + omega*delta) % (2*pi)
    dtheta = theta_k - thetai
    
    sum_term = a * np.exp(-np.square(dtheta / b) * 0.5)
    z = np.sum(sum_term)
    
    return theta, z

def estimate_recg_by_dynamic_model(fs, wave_len, qrs_len,para_switch_point, est_a, est_c ,est_theta):
    # Init Dyn Params
    from math import pi
    phi = np.zeros(6)
    a =  np.zeros(6)
    b = np.zeros(6) + 1e-6

    delta = 1.0/fs
    t = np.arange(0,wave_len)

    RR = fs/119.0
    omega = 2*pi*RR

    arr_theta = [0]
    arr_z = [0]
    para_idx = 0
    for i in t:
        if para_idx < len(para_switch_point) and i == para_switch_point[para_idx]:
            arr_theta[-1] = 0
            RR = fs/qrs_len[para_idx]
            omega = 2*pi*RR
            phi = est_theta[para_idx]
            a = est_a[para_idx]
            b = est_c[para_idx]
            para_idx += 1
        theta, z = ecg_dyn(arr_theta[-1], arr_z[-1], omega, delta, a, phi, b)
        arr_theta.append(theta)
        arr_z.append(z)
    del arr_z[0], arr_theta[0]
    rECG2 = np.array(arr_z)
    arr_theta = np.array(arr_theta)
    return rECG2, arr_theta

# Calculate Metrics
def calculate_metrics(resampled_cleaned_recg, rECG1, rECG2, f_rECG):
    from sklearn.metrics import mean_squared_error, max_error

    mse_rECG1 = mean_squared_error(resampled_cleaned_recg ,rECG1 )
    mse_rECG2 = mean_squared_error(resampled_cleaned_recg ,rECG2 )
    mse_f_rECG = mean_squared_error(resampled_cleaned_recg ,f_rECG )
    corr_rECG1 = np.corrcoef(resampled_cleaned_recg.reshape(-1) ,rECG1 )[0,1]
    corr_rECG2 = np.corrcoef(resampled_cleaned_recg.reshape(-1) ,rECG2 )[0,1]
    corr_f_rECG = np.corrcoef(resampled_cleaned_recg.reshape(-1) ,f_rECG )[0,1]
    
    return mse_rECG1 ,mse_rECG2 ,mse_f_rECG , corr_rECG1 , corr_rECG2 ,corr_f_rECG
    

def calculate_qrs_len(resample_fs, resampled_r_loc):
    RR_intvl = np.diff(resampled_r_loc)
    max_RR = 2 * resample_fs # 2 Sec

    qrs_len = np.where(RR_intvl >= max_RR,np.nan,RR_intvl) # set all rr >= 2 sec to nan for length of qrs complexes
    qrs_len_mean = round(np.nanmean(qrs_len))
    qrs_len = np.where(np.isnan(qrs_len),qrs_len_mean,qrs_len)
    qrs_len = np.insert(qrs_len, 0, qrs_len_mean) # qrs_mean for the length of first qrs complex 
    return qrs_len.astype(int)

@ex.main
def test_and_save(_run, unovis_path, cycles_data_path, lstm_data_path, 
            lstm_model_dir, para_est_model_dir, patient_for_test, is_lstm_normalization):
    save_dir = f'./Direct-Fusion_{now_str}'
    import os
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    test_patient = patient_for_test
    from tensorflow.keras.models import Model, load_model
    import utils
    #Load Model
    para_model = load_model(f'{para_est_model_dir}/model_CNNwRL_test_patient{test_patient}.h5',
                           custom_objects={'GaussLayer':GaussLayer})
    lstm_model = utils.load_model_with_stateful(f'{lstm_model_dir}/model_lstm_test_patient{test_patient}.h5')
        
    from data_loader import load_auto2012, load_qrs_scaler, load_lstm_scaler, patient2rec

    qrs_scaler = load_qrs_scaler(test_patient, cycles_data_path)
    lstm_scaler = load_lstm_scaler(test_patient, lstm_data_path)
    
    record_no = patient2rec[test_patient]
    resample_fs = 250
    
    #Metrics
    mse_vals = []
    corr_vals = []
    print(f'Test Patient: {patient_for_test}')
    for _rec in record_no:
        print(f'Record No.: {_rec}')
        
        cecg, recg, fs, r_loc = load_auto2012(_rec, unovis_path)
        
        print('Begin Filter...')
        # Filtering
        from filtering import supress_motion_artifact, desaturate
        cleaned_cecg, sat_idx = desaturate(cecg, fs, -5, 5)
        cleaned_cecg = supress_motion_artifact(cleaned_cecg, fs)
        cleaned_recg = supress_motion_artifact(recg, fs)
        print('Begin Resample...')
        # Resample
        import scipy.signal
        import utils
        resampled_cleaned_cecg = scipy.signal.resample_poly(cleaned_cecg, resample_fs, fs, axis=0)
        resampled_cleaned_recg = scipy.signal.resample_poly(cleaned_recg, resample_fs, fs, axis=0)
        resampled_r_loc = utils.resample_R_location(r_loc, fs, resample_fs).astype(int)
        sat_idx = utils.resample_R_location(sat_idx, fs, resample_fs).astype(int)
        print('Begin Segment QRS...')
        #Segmentation of QRS Cycles
        qrs_len = calculate_qrs_len(resample_fs, resampled_r_loc)
        
        outlier_idx = np.where(qrs_len <= 0.3 * resample_fs)
        qrs_len = np.delete(qrs_len, outlier_idx)
        resampled_r_loc = np.delete(resampled_r_loc, outlier_idx)

        qrs_compls_cecg = utils.extract_qrs_complex_with_interpolation(resampled_cleaned_cecg,                                                                     resampled_r_loc, qrs_len, 2 * resample_fs)
        qrs_compls_recg = utils.extract_qrs_complex_with_interpolation(resampled_cleaned_recg, resampled_r_loc,
                                                  qrs_len, 2 * resample_fs)
        nqrs_compls_cecg = np.zeros(qrs_compls_cecg.shape)
        for i in range(qrs_compls_cecg.shape[0]):
            nqrs_compls_cecg[i] = qrs_scaler.transform(qrs_compls_cecg[i])
        print('Estimating Morphological Parameter...')    
        # Estimate Morphological Parameter    
        est_recg, est_a, est_theta, est_c = para_model.predict(nqrs_compls_cecg)
        print('Reconstructing by LSTM...')
        # Align the cecg to fit the input size of LSTM
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        import math
        pad_len = math.ceil(resampled_cleaned_cecg.shape[0] / lstm_model.input.shape[1]) * lstm_model.input.shape[1]
        lstm_inputs = pad_sequences(resampled_cleaned_cecg.swapaxes(0,1),
                               padding='post', maxlen=pad_len,value=0,dtype=float).swapaxes(0,1)
        if is_lstm_normalization:
            lstm_inputs = lstm_scaler.transform(lstm_inputs)
        lstm_inputs = np.reshape(lstm_inputs, (-1,*lstm_model.input.shape[1:]))
        
            
        
        rECG1 = lstm_model.predict(lstm_inputs, batch_size=1)
        rECG1 = rECG1.reshape(-1,1)
        rECG1 = rECG1.reshape(-1)
        lstm_model.reset_states()
        
        para_switch_point = utils.calculate_parameter_switch_point(resampled_r_loc, qrs_len)
        
        print('Reconstructing by dynamic model...')
        rECG2, arr_theta = estimate_recg_by_dynamic_model(resample_fs, pad_len, qrs_len,
                                         para_switch_point, est_a, est_c ,est_theta)
        print('Fusion...')
        # Direct Fusion
        f_rECG = (rECG1 + rECG2) / 2

        # Align Data
        rECG2 = rECG2[0:resampled_cleaned_recg.shape[0]]
        rECG1 = rECG1[0:resampled_cleaned_recg.shape[0]]
        f_rECG = f_rECG[0:resampled_cleaned_recg.shape[0]]
        print('Calculate Metrics...')
        mse_rECG1 ,mse_rECG2 ,mse_f_rECG , corr_rECG1 , corr_rECG2 ,corr_f_rECG = calculate_metrics(resampled_cleaned_recg, rECG1, rECG2, f_rECG)
        mse_vals.append([mse_rECG1 ,mse_rECG2 ,mse_f_rECG])
        corr_vals.append([corr_rECG1 , corr_rECG2 ,corr_f_rECG])
        print(f'MSE: {[mse_rECG1 ,mse_rECG2 ,mse_f_rECG]}')
        print(f'CC: {[corr_rECG1 , corr_rECG2 ,corr_f_rECG]}')
        print('Saving Data...')
        import scipy.io
        mdict = {'est_a': est_a, 'est_theta': est_theta, 'est_c': est_c, 
               'rECG1': rECG1, 'rECG2': rECG2, 'f_rECG': f_rECG, 'recg' : resampled_cleaned_recg,
                'mse_rECG1' : mse_rECG1 ,'mse_rECG2':mse_rECG2 ,'mse_f_rECG':mse_f_rECG ,
                 'corr_rECG1':corr_rECG1 , 'corr_rECG2':corr_rECG2 ,'corr_f_rECG':corr_f_rECG,
                'detected_r_loc': resampled_r_loc, 'sat_idx': sat_idx, 'qrs_len': qrs_len,
                'para_switch_point': para_switch_point}
        scipy.io.savemat(f'{save_dir}/direct_fusion_p{test_patient}_rec{_rec}.mat', mdict)
    
    mse_vals = np.array(mse_vals).mean(axis=0)
    corr_vals = np.array(corr_vals).mean(axis=0)
    
    _run.log_scalar("mse.rECG1", float(mse_vals[0]))
    _run.log_scalar("mse.rECG2", float(mse_vals[1]))
    _run.log_scalar("mse.f_rECG", float(mse_vals[2]))
    _run.log_scalar("cc.rECG1", float(corr_vals[0]))
    _run.log_scalar("cc.rECG2", float(corr_vals[1]))
    _run.log_scalar("cc.f_rECG", float(corr_vals[2]))
        
if __name__ == "__main__":
    for i in range(1,7):
        ex.run(config_updates={'patient_for_test': i})