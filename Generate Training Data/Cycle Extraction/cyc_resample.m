% Cycle Resample
clearvars
%% Load Data 
load 'extracted_cycles_cleaned.mat'
resample_fs = 250;

%% Resample

for i = 1:length(extracted_cycles)
    cycles = extracted_cycles(i).cycles;
    
    parfor j = 1: length(cycles)
        all_sigs = cycles(j).data;
        fs = cycles(j).fs
        all_sigs = resample(all_sigs, resample_fs, fs, 'Dimension', 2);
        cycles(j).data = all_sigs;
        cycles(j).fs = resample_fs;
        cycles(j).len = size(all_sigs,2);
    end
    extracted_cycles(i).cycles = cycles;
    
    
end

save('extracted_cycles_resampled.mat', 'extracted_cycles')