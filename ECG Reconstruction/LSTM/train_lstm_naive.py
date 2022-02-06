import numpy as np
import scipy.io

import time


from sacred import Experiment
from sacred.observers import MongoObserver,TelegramObserver

now_str = time.strftime('%Y-%m-%d-%H_%M_%S',time.localtime(time.time()))
ex = Experiment(f'Reconstruction-LSTM-Naive')
ex.observers.append(MongoObserver())
telegram_obs = TelegramObserver.from_config('../telegram_config.json')
ex.observers.append(telegram_obs)

@ex.config
def common_config():
    hidden_states = 50
    network_structure = [
        {'layer': 'lstm', 'units': hidden_states, 'return_sequences': True, },
        {'layer': 'lstm', 'units': hidden_states, 'return_sequences': True,},
        {'layer': 'lstm', 'units': hidden_states, 'return_sequences': True,},
        {'layer': 'dense', 'units': 1},
    ]
    data_path = 'dataset_LSTM_seg_10s_with_recording_nr.mat'
    isNormalization = True
    validation_split = 0.1
    opt = 'adam'
    lr = 1e-5
    epochs = 2500
    early_stop_patience = 15
    early_stop_delta = 1e-3
    patient_for_test = -1
    batch_size = 256
    exp_time = now_str

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
    _run.log_scalar("validation.loss", float(logs.get('val_loss')))

@ex.capture
def load_data_and_normalization(data_path, patient_for_test, isNormalization):
    x_train, y_train, x_test, y_test = None,None,None,None
    
    mat = scipy.io.loadmat(data_path)
    Fs = mat['Fs'][0][0]
    recordings = mat['recordings'][0]
    wave_len = Fs * 5 # 5 sec signal for training
    training_rec = set(range(1,32)) - set(patient2rec[patient_for_test])
    
    train = []
    for i in training_rec:
        idx = i - 1
        numOfseg = recordings[idx][1][0].shape[0]
        for j in range(0, numOfseg):
            train.append(np.expand_dims(recordings[idx][1][0][j][:,0:wave_len], axis=0))
            train.append(np.expand_dims(recordings[idx][1][0][j][:,wave_len:], axis=0))
    train = np.vstack(train)
    train = train.swapaxes(1,2)
    x_train = train[:,:,0:3]
    y_train = train[:,:,3:]
    
    rng = np.random.default_rng(42) # for reproducibility
    shuffle_idx = np.arange(0, x_train.shape[0])
    rng.shuffle(shuffle_idx)
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    test = []
    for i in patient2rec[patient_for_test]:
        idx = i - 1
        numOfseg = recordings[idx][1][0].shape[0]
        for j in range(0, numOfseg):
            test.append(np.expand_dims(recordings[idx][1][0][j][:,0:wave_len], axis=0))
            test.append(np.expand_dims(recordings[idx][1][0][j][:,wave_len:], axis=0))
    test = np.vstack(test)
    test = test.swapaxes(1,2)
    x_test = test[:,:,0:3]
    y_test = test[:,:,3:]
    if isNormalization:
        from sklearn.preprocessing import MinMaxScaler,StandardScaler
        _x = x_train.reshape(-1,3)
        scaler_cecg = StandardScaler()
        x_train = scaler_cecg.fit_transform(_x).reshape(-1,wave_len,3)
        _x = x_test.reshape(-1,3)
        x_test = scaler_cecg.transform(_x).reshape(-1,wave_len,3)
    
    return x_train, y_train, x_test, y_test

@ex.main
def define_and_train(_run, network_structure, validation_split,
              opt, lr, epochs, batch_size,
              early_stop_patience, early_stop_delta,
              patient_for_test):
    
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Input, concatenate
    from tensorflow.keras.layers import Conv1D, MaxPool1D, LSTM, InputLayer, GRU, BatchNormalization
    from tensorflow.keras.models import Model, load_model, Sequential
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard, Callback
    from tensorflow.keras.optimizers import Adam, Adadelta
    
    str2layer = {'lstm': LSTM, 'dense': Dense, 'bn':BatchNormalization,
                'dropout': Dropout}
    str2optimizer = {'adam': Adam}
    
    model_dir = f'./Naive-Model_50S0N'
    import os
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
    ckpt_path = f'{model_dir}/model_lstm_test_patient{patient_for_test}.h5'
    
    class LogPerformance(Callback):
        def on_epoch_end(self, _, logs={}):
            log_performance(logs=logs)
    
    x_train, y_train, x_test, y_test = load_data_and_normalization()
    print(f'x_train{x_train.shape},y_train{y_train.shape},x_test{x_test.shape},y_test{y_test.shape}')
    model = Sequential()
    
    for _layer_cfg in network_structure:
        args = _layer_cfg.copy()
        del args['layer']
        layer = str2layer[_layer_cfg['layer']]
        model.add(layer(**args))
    
    model.compile(loss='mse',
              optimizer=str2optimizer[opt](learning_rate = lr))
    
    
    model_checkpoint_callback = ModelCheckpoint(filepath=ckpt_path, save_weights_only=False,
                        monitor='val_loss', mode='min',save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=early_stop_delta, patience=early_stop_patience, verbose=0, mode='min')
    model.fit(x_train, y_train,
          validation_split=validation_split,
          batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[model_checkpoint_callback, LogPerformance(),early_stop])
    _run.add_artifact(ckpt_path)
    model.load_weights(ckpt_path)
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    return test_loss

if __name__ == "__main__":
    for i in range(1,7):
        ex.run(config_updates={'patient_for_test': i})
