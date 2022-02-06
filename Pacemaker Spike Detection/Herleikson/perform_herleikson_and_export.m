clearvars
load('UnoViS_pacemaker2020.mat')

TP = [];
FP = [];
FN = [];
e_loc = [];
detected_window = 0.050; % half of window
fusion_window = 0.010;  % half of window

fs = 0;
channels = [2 3 4]; 
recordings=struct();
for i = 1:length(unovis)
    ecg_dataset = unovis(i);
    ref_ann_loc = double(extract_pace_pos_from_anns(ecg_dataset.channels(1).ann));

    % CECG1 - 3
    all_ch_locs = {};
    for j = channels
        
        fs = double(ecg_dataset.channels(j).fs);
        w = [detected_window*fs detected_window*fs];
        cecg = double(ecg_dataset.channels(j).data);

        % Filtering
        [cecg_desat,~,~] = desaturation(cecg, fs, 5, -5, 0.5);
        cecg_denoised = cecg_desat - movmedian(cecg_desat,fs*0.150);

        % Perform Herleiksons algorithm
        [vals,detected_locs, ecg_diff] = find_pacespikes_herleikson(cecg_denoised, fs, 0.004,1.0,0.064);
        all_ch_locs{end+1} = detected_locs;
        

        
    end
    % Fuse the detected spikes from different channel in cECG
    fused_locs = three_channel_fusion(all_ch_locs, [fusion_window*fs fusion_window*fs]);
    recordings(i).id = i;
    recordings(i).locs = fused_locs;
        [iTP,iFP,iFN,ie_loc] = check_detected_pos(fused_locs, ref_ann_loc, w);
        TP(end+1) = iTP;
        FP(end+1) = iFP;
        FN(end+1) = iFN;
        e_loc = [e_loc ie_loc];
end


cTP = sum(TP)
cFP = sum(FP)
cFN = sum(FN)
eRMS = mean(e_loc)/fs

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


function ref_loc = extract_pace_pos_from_anns(anns)
ref_loc = [];
    for i = 1:length(anns)
        if strcmp('TA',anns(i).type)% || strcmp('ERR',anns(i).type)
            continue;
        else
            ref_loc(end+1) = anns(i).loc;
        end
    end
end

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
                if (T - t < (50) && (T - t < abs(T - t_prime) || abs(T_prime - t_prime) < abs(T - t_prime)))
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
                if (t - T <= (50) && (t - T < abs(t - T_prime) || abs(t_prime - T_prime) < abs(t - T_prime)))
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
    e_loc = mean((abs(diff)));
end