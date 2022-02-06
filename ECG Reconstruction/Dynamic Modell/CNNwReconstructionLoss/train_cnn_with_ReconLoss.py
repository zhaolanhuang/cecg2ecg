import numpy as np
import scipy.io

import time


from sacred import Experiment
from sacred.observers import MongoObserver, TelegramObserver

now_str = time.strftime('%Y-%m-%d-%H_%M_%S',time.localtime(time.time()))
ex = Experiment(f'Reconstruction-CNNwRL-{now_str}')
ex.observers.append(MongoObserver())
telegram_obs = TelegramObserver.from_config('../telegram_config.json')
ex.observers.append(telegram_obs)


@ex.config
def common_config():
    network_structure = [
        {'layer': 'conv1d', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
        {'layer': 'bn'},
        {'layer': 'activation', 'activation':'relu'},
        {'layer': 'max_pool_1d'},
        {'layer': 'conv1d', 'filters': 32, 'kernel_size': 3, 'padding': 'same'},
        {'layer': 'bn'},
        {'layer': 'activation', 'activation':'relu'},
        {'layer': 'conv1d', 'filters': 64, 'kernel_size': 3, 'padding': 'same'},
        {'layer': 'bn'},
        {'layer': 'activation', 'activation':'relu'},
        {'layer': 'max_pool_1d'},
        {'layer': 'conv1d', 'filters': 64, 'kernel_size': 5, 'padding': 'same'},
        {'layer': 'bn'},
        {'layer': 'activation', 'activation':'relu'},
        {'layer': 'conv1d', 'filters': 64, 'kernel_size': 10, 'padding': 'same'},
        {'layer': 'bn'},
        {'layer': 'activation', 'activation':'relu'},
        {'layer': 'max_pool_1d'},
        {'layer': 'conv1d', 'filters': 80, 'kernel_size': 20, 'padding': 'same'},
        {'layer': 'bn'},
        {'layer': 'activation', 'activation':'relu'},
        
        {'layer': 'global_avg_pooling_1d'},
        {'layer': 'flatten'},
        {'layer': 'dense', 'units': 1024, 'activation': 'relu'},
        {'layer': 'dropout', 'rate' : 0.2},
    ]
    cycles_data_path = 'extracted_cycles_resampled.mat'
    parameters_path = 'cycle_parameters_gauss6_4_0.mat'
    isNormalization = True
    validation_split = 0.2
    opt = 'adam'
    lr = 1e-5
    epochs = 2000
    early_stop_patience = 10
    early_stop_delta = 1e-3
    patient_for_test = -1
    batch_size = 256


patient2rec = dict() # patient id to recording id
patient2rec[1] = [1,2,3,6,24]
patient2rec[2] = [4,7,8,17,21]
patient2rec[3] = [5,9,10,25]
patient2rec[4] = [11,12,13,14,15,26,27,28,29,30,31]
patient2rec[5] = [16,18,19,20]
patient2rec[6] = [22,23]

@ex.capture
def log_performance(_run, logs):
    _run.log_scalar("training.loss", float(logs.get('loss')))
    _run.log_scalar("training.loss.recg", float(logs.get('recg_loss')))
    _run.log_scalar("training.loss.a", float(logs.get('a_loss')))
    _run.log_scalar("training.loss.theta", float(logs.get('theta_loss')))
    _run.log_scalar("training.loss.c", float(logs.get('c_loss')))
    
    _run.log_scalar("validation.loss", float(logs.get('val_loss')))
    _run.log_scalar("validation.loss.recg", float(logs.get('val_recg_loss')))
    _run.log_scalar("validation.loss.a", float(logs.get('val_a_loss')))
    _run.log_scalar("validation.loss.theta", float(logs.get('val_theta_loss')))
    _run.log_scalar("validation.loss.c", float(logs.get('val_c_loss')))
    

@ex.capture
def load_data_and_normalization(cycles_data_path, parameters_path, patient_for_test, isNormalization):
    """
    Load cardiac cycles of cECG and the correponding parameters of rECG
    
    cycles_data_path: the extracted, resampled cECG cycles generated by cyc_resample.m
    parameters_path: the cycles parameters of rECG generated by curve_fit.m
    patient_for_test: Subject ID used for test, so that its data not used for training (leave-one-out-valid)

    Return
    x_train, y_train: training data pair (x_train -> y_train), where x_train is the caridac cycles from cECG,
                      and y_train is the corresponding cycles parameters of rECG [a_i, theta_i, b_i]
    x_test, y_test: test data pair like training data
    """
    resample_Fs = 250
    cyc_data = scipy.io.loadmat(cycles_data_path)
    para_data = scipy.io.loadmat(parameters_path)
    cycles = cyc_data['extracted_cycles'][0]
    paras = para_data['fitting_parameter'][0]
    
    training_rec = set(range(1,32)) - set(patient2rec[patient_for_test])
    
    if isNormalization:
        training_cycles = []
        for i in training_rec: 
            idx = i - 1
            for j in range(0, cycles[idx]['cycles']['data'].shape[1]):
                training_cycles.append(cycles[idx]['cycles']['data'][0,j])
        training_cycles = np.hstack(training_cycles).swapaxes(0,1)
        from sklearn.preprocessing import MinMaxScaler,StandardScaler
        scaler_cecg = StandardScaler()
        scaler_cecg.fit(training_cycles[:, 0:3])
        training_cycles = []
        # normalize all cecgs
        for i in range(0, cycles.shape[0]): 
            for j in range(0, cycles[i]['cycles']['data'].shape[1]):
                seq = cycles[i]['cycles']['data'][0,j].swapaxes(0,1)
                seq[:, 0:3] = scaler_cecg.transform(seq[:, 0:3])
                cycles[i]['cycles']['data'][0,j] = seq.swapaxes(0,1)
                
    intpl_len = 2*resample_Fs   # interpolation length
    x_train, y_train, x_test, y_test = None,dict(),None,dict()
    # Load Training Data
    intpl_cycles = []
    gauss_params = np.array([]).reshape(0,18)  # gauss_params[i] = [a1 ... a5 theta1...theta5 c1 ... c5]
    from scipy.signal import resample
    for i in training_rec: 
        idx = i - 1
        for j in range(0, cycles[idx]['cycles']['data'].shape[1]):
            intpl_seq = resample(cycles[idx]['cycles']['data'][0,j].swapaxes(0,1), 2*resample_Fs)
            intpl_cycles.append(intpl_seq)
        param = np.concatenate((paras[idx]['a'],paras[idx]['b'],paras[idx]['c']), axis=1)
        gauss_params = np.concatenate((gauss_params, param))
    intpl_cycles = np.array(intpl_cycles)
    recg_cycles = intpl_cycles[:,:,3]
    cecg_cycles = intpl_cycles[:,:,0:3]
    
    # Shuffle
    rng = np.random.default_rng(42) # for reproducibility
    shuffle_idx = np.arange(0, gauss_params.shape[0])
    rng.shuffle(shuffle_idx)
    x_train = cecg_cycles[shuffle_idx]
    training_params = gauss_params[shuffle_idx]
    y_train['recg'] = recg_cycles[shuffle_idx]
    y_train['a'] = training_params[:,0:6]
    y_train['theta'] = training_params[:,6:12]
    y_train['c'] = training_params[:,12:18]
    
    # Load Test Data
    intpl_cycles = []
    gauss_params = np.array([]).reshape(0,18)  # gauss_params[i] = [a1 ... a5 theta1...theta5 c1 ... c5]
    from scipy.signal import resample
    for i in patient2rec[patient_for_test]: 
        idx = i - 1
        for j in range(0, cycles[idx]['cycles']['data'].shape[1]):
            intpl_seq = resample(cycles[idx]['cycles']['data'][0,j].swapaxes(0,1), 2*resample_Fs)
            intpl_cycles.append(intpl_seq)
        param = np.concatenate((paras[idx]['a'],paras[idx]['b'],paras[idx]['c']), axis=1)
        gauss_params = np.concatenate((gauss_params, param))
    intpl_cycles = np.array(intpl_cycles)
    recg_cycles = intpl_cycles[:,:,3]
    cecg_cycles = intpl_cycles[:,:,0:3]
    x_test = cecg_cycles
    y_test['recg'] = recg_cycles
    y_test['a'] = gauss_params[:,0:6]
    y_test['theta'] = gauss_params[:,6:12]
    y_test['c'] = gauss_params[:,12:18]
    
    
    return x_train, y_train, x_test, y_test

@ex.main
def define_and_train(_run, network_structure, validation_split,
              opt, lr, epochs, batch_size,
              early_stop_patience, early_stop_delta,
              patient_for_test):
    
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Input, concatenate, Activation
    from tensorflow.keras.layers import Conv1D, MaxPool1D, LSTM, InputLayer, GRU, BatchNormalization, GlobalAveragePooling1D
    from tensorflow.keras.models import Model, load_model, Sequential
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard, Callback
    from tensorflow.keras.optimizers import Adam, Adadelta
    from gauss_layer import GaussLayer
    
    str2layer = {'lstm': LSTM, 'dense': Dense, 'bn':BatchNormalization , 'conv1d': Conv1D,
             'activation': Activation, 'dropout': Dropout, 'flatten': Flatten,
             'global_avg_pooling_1d': GlobalAveragePooling1D, 'max_pool_1d': MaxPool1D}
    str2optimizer = {'adam': Adam}
    ckpt_path = f'model_CNNwRL_test_patient{patient_for_test}_{now_str}.h5'
    
    class LogPerformance(Callback):
        def on_epoch_end(self, _, logs={}):
            log_performance(logs=logs)
    
    x_train, y_train, x_test, y_test = load_data_and_normalization()
   
    inputs = Input(shape=(500,3))    
    x = inputs
    for _layer_cfg in network_structure:
        args = _layer_cfg.copy()
        del args['layer']
        layer = str2layer[_layer_cfg['layer']]
        x = layer(**args)(x)
        
    y_a = Dense(6, activation = 'linear', name='a')(x)
    y_theta = Dense(6, activation = 'linear', name='theta')(x)
    y_c = Dense(6, activation = 'linear', name='c')(x)
    y_params = concatenate([y_a, y_theta, y_c], name='params')
    y_recg = GaussLayer(name='recg')(y_params)
    model = Model(inputs=[inputs], outputs=[y_recg, y_a, y_theta, y_c])
    
    model.compile(loss=['mse', 'mae', 'mae', 'mae'],
              optimizer=str2optimizer[opt](learning_rate = lr))
    
    
    model_checkpoint_callback = ModelCheckpoint(filepath=ckpt_path, save_weights_only=False,
                        monitor='val_loss', mode='min',save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=early_stop_delta, patience=early_stop_patience, verbose=0, mode='min')
    model.fit(x=x_train, y=[ y_train['recg'], y_train['a'], y_train['theta'], y_train['c'] ],
          validation_split=validation_split,
          batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[model_checkpoint_callback, LogPerformance(),early_stop])
    _run.add_artifact(ckpt_path)
    model.load_weights(ckpt_path)
    
    test_loss = model.evaluate(x_test, [ y_test['recg'], y_test['a'], y_test['theta'], y_test['c'] ], verbose=0)
    
    _run.log_scalar("test.loss", float(test_loss[0]))
    _run.log_scalar("test.loss.recg", float(test_loss[1]))
    _run.log_scalar("test.loss.a", float(test_loss[2]))
    _run.log_scalar("test.loss.theta", float(test_loss[3]))
    _run.log_scalar("test.loss.c", float(test_loss[4]))
    
    return test_loss[0]

if __name__ == "__main__":
    for i in range(1,7):
        ex.run(config_updates={'patient_for_test': i})
    