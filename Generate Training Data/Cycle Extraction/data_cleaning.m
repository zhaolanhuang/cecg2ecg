% Outlier Cleaning
clearvars
load 'extracted_cycles_raw.mat'

%% Find and exclude Outlier based on Cycle Length
for i = 1:length(extracted_cycles)
    cycles = extracted_cycles(i).cycles;
    lens = [cycles.len];
    
    mean_len = median(lens)
    std_len = mad(lens,1)
    
    outlier_idx = find(lens > mean_len + 3*std_len);
    cycles(outlier_idx) = [];
    extracted_cycles(i).cycles = cycles;
end



%% Find and exclude Outlier based on Desaturation

for i = 1:length(extracted_cycles)
    cycles = extracted_cycles(i).cycles;
    
    outlier_idx = [];
    
    for j = 1: length(cycles)
        recg = cycles(j).data(end,:);
        if ~isempty(strfind(recg, zeros(1,10)))
            outlier_idx(end+1) = j;
        end
    end
    
    cycles(outlier_idx) = [];
    extracted_cycles(i).cycles = cycles;
    
end

%% Find and exclude Outlier based on R-Peak Pos.
for i = 1:length(extracted_cycles)
    cycles = extracted_cycles(i).cycles;
    outlier_idx = [];
    for j = 1:length(cycles)
        r1pos = round(cycles(j).rr(1) / 3);
        r2pos = round(cycles(j).rr(2) / 3);
        if r1pos <= 10 || r2pos <= 10
            outlier_idx(end+1) = j;
        end
    end
    outlier_idx
    cycles(outlier_idx) = [];
    extracted_cycles(i).cycles = cycles;
end
save('extracted_cycles_cleaned.mat', 'extracted_cycles')