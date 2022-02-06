% Implementation of 1. Berwal D, C.R. V, Dewan S, C.V. J, Baghini MS. Motion Artifact Removal in Ambulatory ECG Signal for Heart Rate Variability Analysis. IEEE Sensors Journal. 2019;19(24):12432-12442. doi:10.1109/JSEN.2019.2939391

function [cleaned_ecg,artifact_signal,original_ecg] = swtlt_artifact_cancellation(ecg, Fs, gr)
    if ~isvector(ecg)
      error('ecg must be a row or column vector');
    end
    if nargin < 3
        gr = 1;   % on default the function always plots
    end
    ecg = ecg(:);

    N = length(ecg);
    M = 7;
    wavelet = 'haar';
    qrs_half_window = Fs * 0.05; % QRS Window = 0.1 s.
    original_ecg = ecg;
    % Fill up the recorded ECG with zeros at the end, so the length can be divided by 2^M
    if mod(N,2^M)~=0 
        x = [ecg; ones(2^M-mod(N,2^M),1)*ecg(end)];
    else
        x = ecg;
    end
    t_zerofill = 0:1/Fs:(length(x)/Fs) - (1/Fs);
    %% Preprocessing
    x_pre = x;
    x = x - movmedian(x, 0.3*Fs); % Median Filter with windows 0.3 s
    [b, a] = butter(4, 150 / (Fs/2)); % low pass filter with fc = 150Hz.
    x = filtfilt(b,a,x);
    
    if gr 
        figure(100)
        plot(t_zerofill,x_pre,'b',t_zerofill,x,'r');
        title('Preprocessing'), hold off
    end
    
    %% First Stage
    SWC = swt(x,M,wavelet); % Compute the wavelet coefficient sequences (SWC(1,:),...,SWC(M,:)) and the scaling coefficient sequence (SWC(M+1,:))
    qrs_pos = qrs_detection(x,Fs);

    for i = qrs_pos
        SWC(1:3,max(1,i - qrs_half_window):min(i + qrs_half_window,length(x))) = 0; % Delete the QRS complexes in the first three coefficient sequences
    end
    
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
    cleaned_ecg_one = x - outlier_signal(:); % Calculate residuals between original ECG and outlier signal to get the cleaned ECG
    
    if gr 
        figure
        tiledlayout(M+1,1);
        for i=1:M+1
            nexttile
            plot(t_zerofill, SWC(i,:),'b',t_zerofill, SWC_new(i,:),'r');
            
        end
        title('Stage 1: SWT')
        figure
        
        plot(t_zerofill, x,'b',t_zerofill, cleaned_ecg_one,'r'), hold on;
        plot(qrs_pos/Fs,qrs_amp,'g*');
        title('Stage 1: ECG')
    end
    
    %% Second Stage
    % Level-Thresholding
    SWC2 = swt(cleaned_ecg_one,M,wavelet);
    SWC2(1:5) = 0;
    
    qrs_peaks = zeros(1,length(cleaned_ecg_one));
    for i = qrs_pos
        qrs_peaks(max(1,i - qrs_half_window):min(i + qrs_half_window,length(x))) = cleaned_ecg_one(max(1,i - qrs_half_window):min(i + qrs_half_window,length(x))); 
    end
    pt_wave = iswt(SWC2, wavelet);
    cleaned_ecg = pt_wave + qrs_peaks;
    cleaned_ecg = filtfilt(b,a, cleaned_ecg);
    cleaned_ecg = cleaned_ecg(1:N);
    artifact_signal = original_ecg' - cleaned_ecg;
    
    t = (0:N-1)/Fs;
    if gr == 1
        figure
        plot(t, original_ecg,'b',t, cleaned_ecg,'r');
        title('Stage 2: ECG')
    end
    
end

function [upper_T,lower_T] = calc_thresholds(SWC,Fs)
    % adopted from MRT-ARTIFACT-CANCELLATION
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
          
        upper_T(n) = med_max + median(abs(med_max - maximum));
        lower_T(n) = med_min - median(abs(med_min - minimum));
    end
end