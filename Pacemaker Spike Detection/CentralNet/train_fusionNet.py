import numpy as np
import scipy.io

import time


from sacred import Experiment
from sacred.observers import MongoObserver, TelegramObserver

now_str = time.strftime('%Y-%m-%d-%H_%M_%S',time.localtime(time.time()))
ex = Experiment(f'Pacemaker-FusionNet')
ex.observers.append(MongoObserver(db_name='Pacemaker'))
telegram_obs = TelegramObserver.from_config('../common/telegram_config.json')
ex.observers.append(telegram_obs)


@ex.config
def common_config():
	network_structure = [
		{'layer': 'conv1d', 'filters': 10, 'kernel_size': 64, 'padding': 'same', 'activation':'relu'},
		{'layer': 'wsum'},
		{'layer': 'bn'},
		{'layer': 'wsum'},
		{'layer': 'max_pool_1d'},
		{'layer': 'wsum'},

		{'layer': 'flatten'},
		{'layer': 'dense', 'units': 100, 'activation': 'relu'},
		{'layer': 'dropout', 'rate' : 0.2},
		{'layer': 'wsum'},
		{'layer': 'dense', 'units': 40, 'activation': 'relu'},
		{'layer': 'dropout', 'rate' : 0.2},
		{'layer': 'wsum'},
	]
	dataset_path = f'../../Dataset/pacemaker_for_cnn3.0.mat'
	isNormalization = False
	validation_split = 0.2 # Portion of training data for validation
	opt = 'adam' # Optimizer
	lr = 1e-4	# Learning rate
	epochs = 1000 # Maximal training epochs
	early_stop_patience = 15
	early_stop_delta = 1e-3
	patient_for_test = -1
	batch_size = 512
	exp_time = now_str
	input_type = 'diff' # diff - only using differential cecg; original - only using cecg; hybrid - using both


@ex.capture
def log_performance(_run, logs):
	"""
	Log the matrics during training
	"""
	_run.log_scalar("training.fusion.loss", float(logs.get('fusion_loss')))
	_run.log_scalar("validation.fusion.loss", float(logs.get('val_fusion_loss')))
	_run.log_scalar("training.fusion.acc", float(logs.get('fusion_accuracy')))
	_run.log_scalar("validation.fusion.acc", float(logs.get('val_fusion_accuracy')))

@ex.capture
def load_data_and_normalization(dataset_path, patient_for_test, isNormalization):
	import sys 
	sys.path.append('..')
	from common import data_loader
	
	return data_loader.load_data_for_cnn(dataset_path, patient_for_test, isNormalization)

@ex.main
def define_and_train(_run, network_structure, validation_split,
			  opt, lr, epochs, batch_size,
			  early_stop_patience, early_stop_delta,
			  patient_for_test,input_type):
	
	import tensorflow as tf
	from tensorflow.keras.utils import to_categorical
	from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Input, concatenate, Activation
	from tensorflow.keras.layers import Conv1D, MaxPool1D, LSTM, InputLayer, GRU, BatchNormalization, GlobalAveragePooling1D
	from tensorflow.keras.models import Model, load_model, Sequential
	from tensorflow.keras import backend as K
	from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard, Callback
	from tensorflow.keras.optimizers import Adam, Adadelta
	from layers import WeightedSum, ApplyUniformLayer
	
	str2layer = {'lstm': LSTM, 'dense': Dense, 'bn':BatchNormalization , 'conv1d': Conv1D,
			 'activation': Activation, 'dropout': Dropout, 'flatten': Flatten,
			 'global_avg_pooling_1d': GlobalAveragePooling1D, 'max_pool_1d': MaxPool1D}
	str2optimizer = {'adam': Adam}
	
	model_dir = f'./Model_{input_type}_{now_str}'
	import os
	if not os.path.isdir(model_dir):
		os.mkdir(model_dir)
	
	ckpt_path = f'{model_dir}/model_fusionNet_test_patient{patient_for_test}.h5'
	
	class LogPerformance(Callback):
		def on_epoch_end(self, _, logs={}):
			log_performance(logs=logs)
	
	x_train, y_train, x_test, y_test = load_data_and_normalization()
   
	if input_type == 'hybrid':
		inputs = tf.keras.Input((129,6))
	else:
		inputs = tf.keras.Input((129,3))
	
	fusion_layer = None
	x = tf.split(inputs, [1 for i in range(inputs.shape[2])], axis=2)

	# Build the CentralNet
	for _layer_cfg in network_structure:
		if _layer_cfg['layer'] == 'wsum':
			if fusion_layer is None:
				fusion_layer = WeightedSum()(x)
			else:
				fusion_layer = WeightedSum()([fusion_layer,*x])
		else:
			args = _layer_cfg.copy()
			del args['layer']
			layer = str2layer[_layer_cfg['layer']]
			x = ApplyUniformLayer(layer, x, **args)
			if fusion_layer is not None:
				fusion_layer = layer(**args)(fusion_layer)
	fusion_layer  = Dense(2, activation = 'softmax')(fusion_layer)
	if input_type == 'diff':
		y_cecg1_diff = Dense(2, activation = 'softmax', name='cecg1_diff')(x[0])
		y_cecg2_diff = Dense(2, activation = 'softmax', name='cecg2_diff')(x[1])
		y_cecg3_diff = Dense(2, activation = 'softmax', name='cecg3_diff')(x[2])
		y_fusion = WeightedSum()([fusion_layer, y_cecg1_diff, y_cecg2_diff, y_cecg3_diff])
		y_fusion = Activation('softmax', name='fusion')(y_fusion)
		outputs = [y_fusion, y_cecg1_diff, y_cecg2_diff, y_cecg3_diff]
	elif input_type == 'original':
		y_cecg1 = Dense(2, activation = 'softmax', name='cecg1')(x[0])
		y_cecg2 = Dense(2, activation = 'softmax', name='cecg2')(x[1])
		y_cecg3 = Dense(2, activation = 'softmax', name='cecg3')(x[2])
		y_fusion  = WeightedSum()([fusion_layer, y_cecg1, y_cecg2, y_cecg3])
		y_fusion = Activation('softmax', name='fusion')(y_fusion)
		outputs = [y_fusion, y_cecg1, y_cecg2, y_cecg3]
	else:
		y_cecg1 = Dense(2, activation = 'softmax', name='cecg1')(x[0])
		y_cecg2 = Dense(2, activation = 'softmax', name='cecg2')(x[1])
		y_cecg3 = Dense(2, activation = 'softmax', name='cecg3')(x[2])
		y_cecg1_diff = Dense(2, activation = 'softmax', name='cecg1_diff')(x[3])
		y_cecg2_diff = Dense(2, activation = 'softmax', name='cecg2_diff')(x[4])
		y_cecg3_diff = Dense(2, activation = 'softmax', name='cecg3_diff')(x[5])
		y_fusion  = WeightedSum()([fusion_layer, y_cecg1, y_cecg2, y_cecg3, y_cecg1_diff, y_cecg2_diff, y_cecg3_diff])
		y_fusion = Activation('softmax', name='fusion')(y_fusion)
		outputs = [y_fusion, y_cecg1, y_cecg2, y_cecg3, y_cecg1_diff, y_cecg2_diff, y_cecg3_diff]
	
	model = Model(inputs=[inputs], outputs=outputs)
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
			  optimizer=str2optimizer[opt](learning_rate = lr))
	if input_type == 'diff':
		x_train = x_train[:,:,3:]
		x_test = x_test[:,:,3:]
	elif input_type == 'original':
		x_train = x_train[:,:,0:3]
		x_test = x_test[:,:,0:3]

	y_train = [y_train for i in range(len(outputs))]
	y_test = [y_test for i in range(len(outputs))]
	
	
	model_checkpoint_callback = ModelCheckpoint(filepath=ckpt_path, save_weights_only=False,
						monitor='val_fusion_loss', mode='min',save_best_only=True)
	early_stop = EarlyStopping(monitor='val_fusion_loss', min_delta=early_stop_delta, patience=early_stop_patience, verbose=0, mode='min')
	model.fit(x=x_train, y=y_train,
		  validation_split=validation_split,
		  batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[model_checkpoint_callback, LogPerformance(),early_stop])
	_run.add_artifact(ckpt_path)
	model.load_weights(ckpt_path)

	test_loss = model.evaluate(x_test, y_test, verbose=0)
	idx = model.metrics_names.index('fusion_loss')
	_run.log_scalar("test.fusion.loss", float(test_loss[idx]))
	idx = model.metrics_names.index('fusion_accuracy')
	_run.log_scalar("test.fusion.acc", float(test_loss[idx]))
	
	return test_loss[idx]

if __name__ == "__main__":
	for i in range(1,21):
		ex.run(config_updates={'patient_for_test': i})
