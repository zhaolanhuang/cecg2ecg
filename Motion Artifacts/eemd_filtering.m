function [cleaned_ecg,artifact_signal,original_ecg] = eemd_filtering(ecg, Fs, gr)
    if ~isvector(ecg)
      error('ecg must be a row or column vector');
    end
    if nargin < 3
        gr = 1;   % on default the function always plots
    end
%     ecg = ecg(:);

    N = length(ecg);
    de_level = 7; % Decompossion level
    ensembles = 10;
    gassian_std = 0.3;
    modes = eemd(ecg, de_level, ensembles, gassian_std);
    if gr
        figure
        stackedplot(modes')
        title('EEMD Modes Level=7')
    end
    
    % Last four Instinct Mode Funtions are seen as motion artifacts.
    for i = 0:3
        modes(end-i,:) = 0;
    end
    
    
    cleaned_ecg = sum(modes,1);
    artifact_signal = ecg - cleaned_ecg;
    original_ecg = ecg;
    
end
