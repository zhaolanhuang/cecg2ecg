% Cycles Extraction

clearvars
load UnoViS_auto2012.mat

window = [1/3 2/3]; % Proprotion of data points before and after R peak, based on RR interval
median_w = 0.150;
resample_fs = 250;

extracted_cycles = struct('id', 0 , 'cycles',[]);
for i = 1:length(unovis)
    cycles = struct('fs', 0 , 'rr',0,'data',[] ,'len', 0);
    all_sigs = [];
    art_idx = [];
    for j = 1:3
        fs = double(unovis(i).channels(j).fs);
        cecg = double(unovis(i).channels(j).data);
	%Filtering
        [cecg,artifact_signal,~] = desaturation(cecg, fs, 5, -5, 0.1, 0);
        cecg = cecg - movmedian(cecg, fs*median_w);
        all_sigs(end+1,:) = cecg;
        art_idx = union(art_idx, find(artifact_signal ~= 0));
    end
    refecg = double(unovis(i).channels(4).data);
    refecg = refecg - movmedian(refecg, fs*median_w);
    all_sigs(end+1,:) = refecg;
    all_sigs(:, art_idx) = 0;
    
    r_locs = double(unovis(i).channels(4).ann(2).loc); % Use manually annoated r peak
    fs = double(unovis(i).channels(4).fs);
    cyc = struct;
    for j = 2:length(r_locs) - 1
        rr_int = [r_locs(j) - r_locs(j-1) r_locs(j+1) - r_locs(j)];

	w = round(rr_int .* window);
        cyc.rr = rr_int;
        cyc.fs = fs;
        cyc.data = all_sigs(: ,r_locs(j) - w(1) : r_locs(j) + w(2));
        cyc.len = size(cyc.data,2);
        cycles(end+1) = cyc;
    end
    cycles(1) = [];
    extracted_cycles(i).id = i;
    extracted_cycles(i).cycles = cycles;
    
end

save('extracted_cycles_raw.mat', 'extracted_cycles')