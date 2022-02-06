% Generate dataset for training, validation and test

clearvars
% percentage traning, validation data
Fs = 1000;
seg_len = [64 64]; %total 129 samples;

load('UnoViS_pacemaker2020.mat')

channels = [2 3 4];
recordings = struct(); % Group the segments by their recording nr.
rng(42); % for reproducity
for i = 1:length(unovis)
    ecg_dataset = unovis(i);
    ref_ann_loc = double(extract_pace_pos_from_anns(ecg_dataset.channels(1).ann));
    cecg = [ecg_dataset.channels(channels).data];
    cecg = cecg';
    cecg = cecg - movmedian(cecg, Fs*0.150, 2);
    cecg_diff = [];
    
    positive = {};
    negative = {};
    for j = 1:3
        cecg_diff(end+1,:) = filter([1 1 -1 -1],[1],cecg(j,:));
    end
    
    % generate positive samples
    for loc = ref_ann_loc
        pos_seg = cecg(:,max(1,loc - seg_len(1)): min(end, loc + seg_len(2)));
        pos_seg = [pos_seg ; cecg_diff(:,max(1,loc - seg_len(1)): min(end, loc + seg_len(2)))];
        positive{end+1} = pos_seg;
    end
    
    % generate negative samples
    N = length(cecg(1,:));
    % 10 ms away from spikes annotation
    spikes_region = arrayfun(@(x) max(1, x - 10):min(N, x + 10), ref_ann_loc, 'UniformOutput', false);
    spikes_region = cell2mat(spikes_region);
    spikes_region = unique(spikes_region);
    for j = 65:10:N - 64
        if isempty(find(spikes_region == j, 1)) % Skip the region nearby pacemaker spikes
            neg_seg = cecg(:,max(1,j - seg_len(1)): min(end, j + seg_len(2)));
            neg_seg = [neg_seg ; cecg_diff(:,max(1,j - seg_len(1)): min(end, j + seg_len(2)))];
            negative{end+1} = neg_seg;
        end
    end
    % Balance the positive and negative samples
    pos_len = length(positive);
    if pos_len ~= 0 
        sel_neg_idx = randsample(1:length(negative),pos_len);
    else
        sel_neg_idx = randsample(1:length(negative),20);
    end
    negative = negative(sel_neg_idx);
    
    recordings(i).id = i;
    recordings(i).positive = positive;
    recordings(i).negative = negative;
end


function ref_loc = extract_pace_pos_from_anns(anns)
ref_loc = [];
    for i = 1:length(anns)
        if strcmp('TA',anns(i).type) % || strcmp('ERR',anns(i).type) 
            continue;
        else
            ref_loc(end+1) = anns(i).loc;
        end
    end
end
