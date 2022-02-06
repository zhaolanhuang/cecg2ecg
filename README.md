# Pattern Recognition in the Capacitive Electrocardiogram and Reconstruction of the Reference Electrocardiogram

Author: Zhaolan Huang

The present thesis develops a framework for the reconstruction of the reference electrocardiogram (ECG) from the three-channel capacitive ECG, and the detection of pacemaker spikes in the capacitive ECG. It explores and compares artificial neural networks with different structures on signal reconstruction and pattern recognition.

This framework contains two data paths, where the first path utilizes Long Short-Term Memory (LSTM) for direct estimation of the reference ECG and the second uses a dynamic model to generate noise-free reconstruction. The parameters of the dynamic model are estimated by a convolutional neural network. The two data paths are averaged with or without weights to form the final reconstruction output. The results show that neural networks are capable of reconstructing the reference ECG.
The classifier of pacemaker spikes is implemented based on a neural network. Different stimulation modes of pacemakers are grouped to find the optimal working points of the classifier. 

Keywords: Capacitive Electrocardiogram, Artificial Neural Network, ECG Reconstruction, Pacemaker Spike, Deep Learning

## Requirements

The code is tested with Python Version 3.7.0-64 and MATLAB R2020b on Windows 10 and runs fastest with a cuda enabled graphics card with Cuda 10.1. If you want to use a gpu install the appropriate tensorflow package version 2.7.0.

The training of neural network is conducted under the experiment management tool Sacred. It needs MongoDB to record the experiment results.

## Package Dependence

- numpy
- scipy
- sacred
- tensorflow
- sklearn

## Structure

├─ECG Reconstruction				Neural networks for ECG Reconstruction
│  ├─Dynamic Modell					Three different neural networks for ECG model identification
│  │  ├─CNNwReconstructionLoss		CNN with ReconstructionLoss
│  │  ├─Linear Regression			Naive model
│  │  ├─LSTM-FCN					LSTM connected with FCN
│  │  └─MultiCNN					Duplicated CNNs for different types of parameter
│  ├─Fusion							Fuse the outputs from dynamic system and LSTM
│  │  ├─Direct Fusion 				Average the outputs
│  │  └─SQI Fusion 					Weighted average based on SQI
│  └─LSTM							LSTM for direct ECG reconstruction from capacitive ECG
│			
├─Generate Training Data 			Preprocessing data for training Neural networks
│
├─Motion Artifacts					Evaluate different methods for suppressing motion artifacts
│
├─Pacemaker Spike Detection			Evaluate two methods for detection of pacemaker spikes
│  ├─CentralNet						Implementation of a special CNN with fusion structure
│  │
│  └─Herleikson						Implementation of Herleiksons algorithm
│
└─QRS Detection

The files "train_*.py" can be directly run for training neural network.