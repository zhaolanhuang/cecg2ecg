% DENOISING  denoise an ECG recording using the wavelet shrinkage method by Donoho and Johnstone
%
% The algorithm denoise an ECG recording the the wavelet shrinkage method
% by Donoho and Johnstone (Ideal spatial adaptation via wavelet shrinkage,
% 1994). It also reduces the R peak. Hence, its primarily used for the QRS
% detection.
%
% [ecg_denoised] = denoising(ecg,M)
%
% ecg - recorded ECG signal
% M - highest wavelet stage
%
% ecg_denoised - denoised ECG signal
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ecg_denoised,T,sigma] = denoising(ecg,M)

    N = length(ecg);
    
    % Fill up the recorded ECG with zeros at the end, so the length can be divided by 2^M
    if mod(N,2^M)~=0
        x = [ecg; zeros(2^M-mod(N,2^M),1)];
    else
        x = ecg;
    end
    
    SWC = swt(x,M,'haar'); % Perform the stationary wavelet transform (SWT) using the Haar wavelet as mother wavelet
    
    SWC_new = zeros(M+1,length(x));
    
    for i=1:M
        sigma(i) = 1.4826*mad(abs(SWC(i,:))); % Calculate the the normalized median absolute deviation (robust standard deviation) of the coefficients for each stage
        T(i) = 4*sigma(i); % Compute the threshold for each stage
        
        temp = find(abs(SWC(i,:)>=T(i))); % Perform hard thresholding, cancel out coefficients smaller than abs(T)  
        SWC_new(i,temp) = SWC(i,temp);
    end
    SWC_new(M+1,:) = SWC(M+1,:);    
    
    ecg_denoised = iswt(SWC_new,'haar'); % Perform inverse SWT to get the denoised ECG signal
end
    
    