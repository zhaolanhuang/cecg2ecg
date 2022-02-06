% QRS_DETECTION  Detects QRS complexes in a ECG recording corrupted by motion induced artifact
%
% The algorithm uses the proposed method by Chen et al. (A real-time QRS detection method based on moving-averaging
% incorporation with wavelet denoising, 2006). We robustified the method
% agianst motion induced artifacts. 
% First, the recording is denoised by the wavelet shrinkage method developed by
% Donoho and Johnstone. Then, the P waves, T waves and artifacts are
% suppressed by an high-pass. After that, the qrs feature signal is
% computed, which represents the energy over a window. At the end, the QRS
% complexes are detected by a threshold, which we proposed to robustified
% the method against motion induced artifacts.
%
% [qrs_pos] = qrs_detection(ecg,Fs)
%
% ecg - recorded ecg signal
% Fs - sampling frequency in Hz of the recorded ecg signal
%
% qrs_pos - the detected QRS positions
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [qrs_pos] = qrs_detection(ecg,Fs)

    N = length(ecg); % length of the ECG recording
    L = 5; % the length of the high-pass
    K = 15; % the window length to compute the feature signal
    c = 2; % constant for threshold calculation  
    M = find_M(Fs); % see additional_info_default_param.pdf for more info                                                
    
    denoised_ecg = denoising(ecg,M); % Perform the denoising method
    
    % Filter the denoised signal with an high-pass
    y = zeros(1,N-floor(L/2));
    for n=ceil(L/2):N-floor(L/2)
        y2=mean(denoised_ecg(n-floor(L/2):n+floor(L/2)));
        y1=denoised_ecg(n);
        y(n)=y1-y2;
    end
    
    % Compute the QRS feature signal
    z = zeros(1,length(y)-floor(K/2));
    for n=ceil(K/2):length(y)-floor(K/2)
        z(n)=sum(y(n-floor(K/2):n+floor(K/2)).^2);
    end
    
    % Split z(n) in segments and calculate the maximum of each segment
    l=1;
    for i=1:Fs:length(z)-Fs
        l_max(l)=max(z(i:i+Fs-1));
        l=l+1;
    end
    
    
    med_lm = median(l_max); % Compute the median (robust mean) of all maxima
    mad_lm = 1.4826*mad(l_max); % Compute the normalized median absolute deviation (robust standard deviation) of all maxima 
    T = med_lm - c*mad_lm; % Compute the threshold
    
    qrs_pos = find(z >= T); % find the QRS comlesxes
end