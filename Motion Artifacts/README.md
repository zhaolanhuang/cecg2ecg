Requirements

The code is tested with Python Version 3.7.0-64 and Matlab R2020b on Windows 10. 
Some MATLAB  codes were mixed-programmed with Python. It requires following packages:
- sklearn
- numpy
- scipy
- biosppy
- pywt

Description

ma_denoised_qrs_performance.m: Evaluate the Performance of different algorithms
eemd_filtering.m: Implementation of EEMD-based Filtering
eemd_matlab: Implementation of EEMD algorithm, taken from https://github.com/leeneil/eemd-matlab
multi-resolution-thresholding: MRT algorithm, taken from Dr. Michael Muma's paper.
OSEA: OSEA package for QRS Detection, based on Pan-Tompkins Algorithm
SWT-LT: Implementation of SWT-LT algorithm
WaveletICA: Implementation of WaveletICA algorithm