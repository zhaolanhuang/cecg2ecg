% CALC_THRESHOLDS  calculates the thresholds for the multi-resolution thresholding algorithm
%
% The method calculates an upper and a lower threshold for each coefficient
% sequenced obtained by the stationary wavelet transform (SWT). First, each
% sequence is splitted into segments. Then, for each segment a maximum and
% a minimum are computed. With all maxima and minima of each sequence, we
% calculate an upper and a lower threshold for each sequence.
%
% [upper_T,lower_T] = calc_thresholds(SWC,Fs)
%
% SWC - matrix containing in each row the coefficient sequences obtained by the SWT
% Fs - sampling frequency in Hz of the recorded ecg signal
%
% upper_T - the calculated upper thresholds for each sequence
% lower_T - the calculated lower thresholds for each sequence

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [upper_T,lower_T] = calc_thresholds(SWC,Fs)

    upper_T = zeros(size(SWC,1),1);
    lower_T = zeros(size(SWC,1),1);
    
    [N,I] = size(SWC);
    
    % Calculation is performed for each sequence separately
    for n=1:N
        l=1;
        % Split sequence into segments and find for each segment a maximum and a minimum
        for i=1:Fs:I-Fs
            maximum(l)=max(SWC(n,i:i+Fs-1));
            minimum(l)=min(SWC(n,i:i+Fs-1));
            l=l+1;
        end
        
        med_max = median(maximum); % Compute the median (robust mean) of all maxima
        mad_max = 1.4826*mad(maximum,1); % Compute the normalized median absolute deviation (robust standard deviation) of all maxima 
        
        med_min = median(minimum); % Compute the median (robust mean) of all minima
        mad_min = 1.4826*mad(minimum,1); % Compute the normalized median absolute deviation (robust standard deviation) of all minima 
    
        % Calculate a c_max and a c_min
        c_max = med_max+mad_max;
        c_min = med_min-mad_min;
    
        % Define thresholds: upper_T is the highest maximum below c_max; lower_T is the lowest minimum above c_min
        if isempty(find(maximum >c_max, 1))
            upper_T(n) = c_max;
        else
            upper_T(n) = max(maximum(maximum < c_max));
        end
    
        if isempty(find(minimum < c_min, 1))
            lower_T(n) = c_min;
        else
            lower_T(n) = min(minimum(minimum > c_min));
        end
    end
end