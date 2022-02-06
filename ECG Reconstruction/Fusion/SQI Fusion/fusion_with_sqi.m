clearvars
addpath .
cd Direct-Fusion-with-QRS-Dect\
load UnoViS_auto2012.mat
resample_fs = 250;
%% Fusion
cc_vals = [];
rms_vals = [];
for pat = 1:6
    mat_files = {dir(['direct_fusion_p' num2str(pat) '*']).name};
    recgs = [];
    recg1s = [];
    recg2s = [];
    f_recgs = [];
    for i = 1:length(mat_files)
        fname = mat_files{i};
        load(fname);
        rec_no = regexp(fname,'(?<=rec)\d*','match');
        rec_no = str2num(rec_no{1});
        recg = recg';
        [cecgs, iSQIs, vSQIs] = filtering_and_calculate_sqi(unovis, rec_no, resample_fs);

        f_sqi_recg = sqi_fusion(recg, rECG1, rECG2, iSQIs, vSQIs);
        recgs = [recgs recg];
        recg1s = [recg1s rECG1];
        recg2s = [recg2s rECG2];
        f_recgs = [f_recgs f_sqi_recg];
    end
    cc_recg1 = corrcoef(recg1s, recgs);
    cc_recg2 = corrcoef(recg2s, recgs);
    cc_frecg = corrcoef(f_recgs, recgs);
    
    rms_recg1 = rms(recg1s-recgs);
    rms_recg2 = rms(recg2s-recgs);
    rms_frecg = rms(f_recgs-recgs);
    
    cc_vals = [cc_vals; cc_recg1(2) cc_recg2(2) cc_frecg(2)];
    rms_vals = [rms_vals; rms_recg1 rms_recg2 rms_frecg];
end



function [iSQIs, vSQIs] = cecgs_sqi(cecgs, fs)
sqis = [];
    for j = 1:3
        [iSQIs, vSQIs] = moving_ecg_sqi(cecgs(j,:), fs);
        sqis(end+1,:) = vSQIs;
    end
vSQIs = harmmean(sqis, 1) ;

end

function [cecgs, iSQIs, vSQIs] = filtering_and_calculate_sqi(unovis, rec_no, resample_fs)
median_w = 0.150;
cecgs = [];
art_idx = [];
for j = 1:3
    fs = double(unovis(rec_no).channels(j).fs);
    cecg = double(unovis(rec_no).channels(j).data);
    [cecg,artifact_signal,~] = desaturation(cecg, fs, 5, -5, 0.1, 0);
    cecg = cecg - movmedian(cecg, fs*median_w);
    cecgs(end+1,:) = cecg;
    art_idx = union(art_idx, find(artifact_signal ~= 0));
end
cecgs(:, art_idx) = 0;
cecgs = resample(cecgs, resample_fs, fs, 'Dimension', 2);
[iSQIs, vSQIs] = cecgs_sqi(cecgs, resample_fs);
end


function f_recg = sqi_fusion(recg, rECG1, rECG2, iSQIs, vSQIs)
f_recg = recg;
window = round(250*1);
    for i = 1:length(iSQIs)
        snippet = rECG1(iSQIs(i):iSQIs(i)+window) * vSQIs(i) + rECG2(iSQIs(i):iSQIs(i)+window) * (1-vSQIs(i));
        f_recg(iSQIs(i):iSQIs(i)+window) =snippet;
    end
end