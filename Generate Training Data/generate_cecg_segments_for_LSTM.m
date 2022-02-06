% Gerenate 10 sec segments(cecg1-3 RefEcg) from dataset, resample to 250Hz
% make saturated segments to zeros

%Use UnoVis Auto 2012, Amp(R-peak) = 0.15mv?
clearvars
load UnoViS_auto2012.mat

sat = 5;

resample_fs = 250;

median_w = 0.150; % Median Filter Window

segment_length = 10; %10 sec.
recordings = struct('id', 0 , 'segments',{}); % Group the segments by their recording nr.
for i = 1:length(unovis)
    % cecg 1-3
    segments = {};
    art_idx = [];
    all_sigs = [];
    for j = 1:3
        fs = double(unovis(i).channels(j).fs);
        cecg = double(unovis(i).channels(j).data);
        % Filtering
        [cecg,artifact_signal,~] = desaturation(cecg, fs, sat, -sat, 0.1, 0);
        cecg = cecg - movmedian(cecg, fs*median_w);
        all_sigs(end+1,:) = cecg;
        art_idx = union(art_idx, find(artifact_signal ~= 0));
    end
    recg = double(unovis(i).channels(4).data);
    all_sigs(end+1,:) = recg - movmedian(recg, fs*median_w);
    all_sigs(:, art_idx) = 0;
   

    rloc_4 = unovis(i).channels(4).ann(2).loc;
    
    for j = 1:segment_length*fs:size(all_sigs,2) - segment_length*fs
        iBegin = j;
        iEnd = j+segment_length*fs-1;
        
        % Exclude outliers if a 10s-segment contains less than 5 R-peaks
        if  length(find(iBegin<rloc_4 & rloc_4<iEnd)) < 5
            continue;
        end
        
        sig = all_sigs(:,iBegin:iEnd);
        sig = resample(sig, resample_fs, fs, 'Dimension', 2);
        segments{end+1} = sig;
    end
    recordings(i).id = i;
    recordings(i).segments = segments;
end
Fs = resample_fs;
save('dataset_LSTM_seg_10s_with_recording_nr.mat', 'recordings', 'Fs')
