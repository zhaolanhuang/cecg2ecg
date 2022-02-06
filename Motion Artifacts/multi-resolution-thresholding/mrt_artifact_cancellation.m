% MRT_ARTIFACT_CANCELLATION  Cancels out motion induced artifact in an ECG recording using Multi-Resolution Thresholding (MRT)
%
% Algorithm estimates the outlier/artifact signal and subtracts it from the
% recorded signal (cleaned_ecg = original_ecg - outlier_signal). 
% Therefore, the stationary wavelet transform (swt), a QRS detection method 
% (Chen et al. 2006) robustified against artifacts and multi-resolution 
% thresholding are used.
%
% [cleaned_ecg,original_ecg,outlier_signal] = mrt_cancelation(ecg,Fs,M,wavelet)
%
% ecg - recorded ecg signal
% Fs - sampling frequency in Hz of the recorded ecg signal
% M - highest wavelet stage (Default: 5)
% wavelet - the used mother wavelet (character string) (Default: Haar wavelet)
%           available mother wavelets can be found in the Matlab Product Help
%
% cleaned_ecg - the cleaned ECG (same length as ecg)
% original_ecg - the recorded ECG filled up with zeros
% outlier_signal - estimated outlier/artifact signal (same length as original_ecg)
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [cleaned_ecg,original_ecg,outlier_signal] = mrt_artifact_cancellation(ecg,Fs,M,wavelet)


    if nargin < 4 
        wavelet = 'Haar';
    end
    if nargin < 3
        M = find_M(Fs); % see additional_info_default_param.pdf for more info                                                
    end
   
    N = length(ecg);
    
    % Fill up the recorded ECG with zeros at the end, so the length can be divided by 2^M
    if mod(N,2^M)~=0 
        x = [ecg; zeros(2^M-mod(N,2^M),1)];
    else
        x = ecg;
    end
  
    
    SWC = swt(x,M,wavelet); % Compute the wavelet coefficient sequences (SWC(1,:),...,SWC(M,:)) and the scaling coefficient sequence (SWC(M+1,:))
    
    qrs_pos = qrs_detection(x,Fs); % Perform the QRS complex detection
    SWC(1:3,qrs_pos) = 0; % Delete the QRS complexes in the first three coefficient sequences
    
    [upper_T,lower_T] = calc_thresholds(SWC,Fs); % Calculate a upper and a lower threshold for each stage (coefficient sequence)
    
    % Estimate coefficient sequences which represent the outliers/artifacts by hard thresholding
    SWC_new = zeros(size(SWC));
    for i=1:M+1
        temp=find((SWC(i,:) > upper_T(i))|(SWC(i,:) < lower_T(i)));
        if ~isempty(temp)
            SWC_new(i,temp) = SWC(i,temp); % only coefficients outside the thresholds are considered
        end
    end
    
    outlier_signal = iswt(SWC_new,wavelet); % Perform the inverse SWT to get the estimated outlier/artifact signal
    original_ecg = x; % Original ECG filled up with zeros
    cleaned_ecg = original_ecg - outlier_signal'; % Calculate residuals between original ECG and outlier signal to get the cleaned ECG 
    cleaned_ecg = cleaned_ecg(1:N); % Cleaned ECG with lentgh of input ECG
end
    
    
    