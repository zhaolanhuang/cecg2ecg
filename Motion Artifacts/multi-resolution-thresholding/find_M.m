function M = find_M(Fs)
% Finds the optimal value of M based on default Sampling Frequency
% 
% Inputs:
%   Fs      Sampling frequency
%
% Outputs:
%   M       Number of wavelet stages
% 
% Example: 
%   M = find_M(320);
% 
% This calculation of M is based on experiments performed to minimize
% mean square error(MSE) between original signal and cleaned signal. 
% For more details on default choice of M based on Fs, 
% see additional_info_default_param.pdf
if Fs <= 64 
        M = 3;
    elseif Fs > 64 && Fs <= 320
        M =5;
    elseif Fs > 320 && Fs <= 640
        M = 6;
    else M = 7;
end   
    