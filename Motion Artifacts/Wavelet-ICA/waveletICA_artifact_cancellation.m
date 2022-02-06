% Implementation of 
%1. Abbaspour S, Gholamhosseini H, Linden M. Evaluation of Wavelet Based Methods in Removing Motion Artifact from ECG Signal. In: Mindedal H, Persson M, eds. 16th Nordic-Baltic Conference on Biomedical Engineering. IFMBE Proceedings. Springer International Publishing; 2015:1-4. doi:10.1007/978-3-319-12967-9_1
function [cleaned_ecg,artifact_signal,original_ecg] = waveletICA_artifact_cancellation(ecg, Fs, gr)
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
    original_ecg = ecg;
    
    % Fill up the recorded ECG with zeros at the end, so the length can be divided by 2^M
    if mod(N,2^M)~=0 
        x = [ecg; ones(2^M-mod(N,2^M),1)*ecg(end)];
    else
        x = ecg;
    end
    t_zerofill = 0:1/Fs:(length(x)/Fs) - (1/Fs);
    SWC = swt(x,M,wavelet); % Compute the wavelet coefficient sequences (SWC(1,:),...,SWC(M,:)) and the scaling coefficient sequence (SWC(M+1,:))
   
    if gr 
        figure
        tiledlayout(M+1,1);
        for i=1:M+1
            nexttile
            plot(t_zerofill, SWC(i,:),'b');
        end
        title('Stage 1: SWT')
    end
    
    % Perform ICA on stationary wavelet components
    [icasig, A, ica_obj] = fastICA_py(SWC', M+1);
    icasig = icasig';
    % Calculate Variance robustly
    for j = 1:M+1
        seg_var = [];
        for i=1:Fs:length(icasig(1,:))-Fs
            seg_var(end+1) = 1.4826* mad(icasig(j,i:i+Fs-1),1);
        end
        var_of_component(j) = 1.4826* mad(seg_var,1);
    end

    mean_of_var = mean(var_of_component);
    std_of_var = std(var_of_component);
    idx = find(var_of_component > mean_of_var + std_of_var);
    b = fir1(100, 10/(Fs/2), 'high');
    icasig_new = icasig;

    for i = 1:length(idx)
        icasig_new(idx(i),:) = filtfilt(b,1,icasig_new(idx(i),:));
    end
    
    if gr 
        figure
        tiledlayout(M+1,1);
        for i=1:M+1
            nexttile
            plot(t_zerofill, icasig(i,:),'r', t_zerofill,icasig_new(i,:),'b--');
            xlim([t_zerofill(1) t_zerofill(end)])
        end
        legend('Orig.','Filtered')
%         title('Stage 1: ICA and Filtering')
    end
    
    SWC_new = ica_obj.inverse_transform(icasig_new');
    SWC_new = double(SWC_new);
    SWC_new = SWC_new';
    cleaned_ecg = iswt(SWC_new, wavelet);
    cleaned_ecg = cleaned_ecg(1:N);
    cleaned_ecg = cleaned_ecg - median(cleaned_ecg);

    original_ecg = ecg';
    artifact_signal = original_ecg - cleaned_ecg;
    if gr 
        figure
        plot(original_ecg, 'b'), hold on
        plot(cleaned_ecg,'r')
    end
    
end