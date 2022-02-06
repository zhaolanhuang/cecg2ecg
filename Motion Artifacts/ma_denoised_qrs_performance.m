clearvars
load('UnoViS_bed2013.mat')
addpath eemd-matlab
addpath multi-resolution-thresholding
addpath OSEA
addpath SWT-LT
addpath Wavelet-ICA
TP = [];
FP = [];
FN = [];
e_loc = [];
detected_window = 0.150;
fusion_window = 0.050;  % half of window
fs = 0;
exc_time = 0;
switch_value = 0; % Switch the algorithms
% 0 - NO Denoised 1 - Median Filter 2 - MRT 3- SWT-LT 4 - WaveletICA 5 - EEMD
for i = 1:length(unovis)
    ecg_dataset = unovis(i);
    
    % CECG1 - 3
    all_ch_locs = {};
    % Reference annotations from rECG
    man_ann_loc = double(ecg_dataset.channels(7).ann(2).loc);
    for j = 1:3
        
        fs = double(ecg_dataset.channels(j).fs);
        w = [detected_window*fs detected_window*fs];
        cecg = double(ecg_dataset.channels(j).data);

        cecg_desat = cecg;
        tic
        if switch_value == 0
            cecg_denoised = cecg_desat;
        elseif switch_value == 1
             cecg_denoised = cecg_desat - movmedian(cecg_desat,fs*0.150);
        elseif switch_value == 2
            [cecg_denoised_mrt,~,~] = mrt_artifact_cancellation(cecg_desat,fs);
            cecg_denoised = cecg_denoised_mrt';
        elseif switch_value == 3
            [cecg_denoised,~,~] = swtlt_artifact_cancellation(cecg_desat', fs, 0);
        elseif switch_value == 4
            [cecg_denoised, ~, ~] = waveletICA_artifact_cancellation(cecg_desat, fs, 0);
        elseif switch_value == 5
            [cecg_denoised,~,~] = eemd_filtering(cecg_desat', fs, 1);
        end
        exc_time = exc_time + toc;

        % Use OSEA Package to perform R-peaks detection
         [detected_loc, typeCode] = findRPeaks(cecg_denoised, 500, fs);
    all_ch_locs{end+1} = detected_loc;
    end
    fused_locs = three_channel_fusion(all_ch_locs, [fusion_window*fs fusion_window*fs]);
        [iTP,iFP,iFN,ie_loc] = check_detected_pos(fused_locs, man_ann_loc, w);
        TP(end+1) = iTP;
        FP(end+1) = iFP;
        FN(end+1) = iFN;
        e_loc = [e_loc ie_loc/fs];
end
cTP = sum(TP)
cFP = sum(FP)
cFN = sum(FN)
eRMS = mean(e_loc)
exc_time

% Fusion the detected locations from different channels
function fused_locs = three_channel_fusion(all_ch_locs, fusion_window)
fused_locs = [];
for i = 1:length(all_ch_locs)
    locs = all_ch_locs{i};
    for j = 1:length(locs)
        idx = find(locs(j) - fusion_window(1) < fused_locs & fused_locs < locs(j) + fusion_window(2));
        if isempty(idx)
            fused_locs(end+1) = locs(j);
        else
            % If two locations are close to each other (determined by
            % fusion_window), then use their average.
            fused_locs(idx) = round(mean([fused_locs(idx) locs(j)]));
        end
    end
end
fused_locs = sort(fused_locs);
end

% Calculate Classification Metrics
% detected_loc: locations from detector
% ref_loc: reference locations from annotation in data set
% w: tolerance window between detected and reference locations
function [TP,FP,FN,e_loc] = check_detected_pos(detected_loc, ref_loc, w)
    ref = ref_loc;
    test = detected_loc;

    TP = 0;
    FP = 0;
    FN = 0;
    i = 1;
    j = 1;

    diff = [];

    while (i < length(ref) && j < length(test))
        t = test(j);
        T = ref(i);
        if (j ~= length(test) && i ~= length(ref))
            t_prime = test(j + 1);
            T_prime = ref(i + 1);
            if (t < T)
                if (T - t < w(1) && (T - t < abs(T - t_prime) || abs(T_prime - t_prime) < abs(T - t_prime)))
                    % match a and A
                    TP = TP + 1;
                    % get next t
                    j = j + 1;
                    % get next T
                    i = i + 1;

                    diff(end+1) = (T - t);
                else
                    % no match for t
                    FP = FP + 1;
                    % get next t
                    j = j + 1;
                end
            else
                if (t - T <= w(2) && (t - T < abs(t - T_prime) || abs(t_prime - T_prime) < abs(t - T_prime)))
                    % match a and A
                    TP = TP + 1;
                    % get next t
                    j = j + 1;
                    % get next T
                    i = i + 1;

                    diff(end+1) = (T - t);

                else
                    % no match for T
                    FN = FN + 1;
                    % get next T
                    i = i + 1;
                end
            end
        end
    end
    FN = max(0, length(ref_loc) - TP);
    e_loc = mean(abs(diff));
end