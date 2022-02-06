function [cleaned_ecg,artifact_signal,original_ecg] = desaturation(ecg, Fs, upper_limit, lower_limit, reflation, gr)
    x = ecg(:);
    N = length(x);
    if nargin < 6
        gr = 1;   % on default the function always plots
    end
    if nargin < 5
        reflation = 0.1; % default: 0.1s reflation
    end
    rf_pts = reflation * Fs;
    
    up_idx = find(x >= upper_limit);
    lw_idx = find(x <= lower_limit);
    
    artifact_idx = arrayfun(@(x) max(1, x - rf_pts):min(N, x + rf_pts), [up_idx' lw_idx'], 'UniformOutput', false);
    artifact_idx = cell2mat(artifact_idx);
    artifact_idx = unique(artifact_idx);
    artifact_signal = zeros(N,1);
    artifact_signal(artifact_idx) = x(artifact_idx);
    cleaned_ecg = x - artifact_signal;
    original_ecg = x;

end

