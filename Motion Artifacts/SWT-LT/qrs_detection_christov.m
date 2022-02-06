function [qrs_amp_raw,qrs_i_raw]= qrs_detection_christov(ecg,fs,gr)

%% Inputs
% ecg : raw ecg vector signal 1d signal
% fs : sampling frequency e.g. 200Hz, 400Hz and etc
% gr : flag to plot or not plot (set it 1 to have a plot or set it zero not
% to see any plots
%% Outputs
% qrs_amp_raw : amplitude of R waves amplitudes
% qrs_i_raw : index of R waves

if ~isvector(ecg)
  error('ecg must be a row or column vector');
end
if nargin < 3
    gr = 1;   % on default the function always plots
end
ecg = ecg(:); % vectorize



fullpath = mfilename('fullpath'); 
[path,~] = fileparts(fullpath);
insert(py.sys.path,int32(0),path);
py.importlib.import_module('ecgdetectors');

detector = py.ecgdetectors.Detectors(fs);
idx = detector.christov_detector(py.list(ecg));
qrs_i_raw = cellfun(@double,cell(idx));
qrs_amp_raw = ecg(qrs_i_raw);

if gr == 1
    N = length(ecg);
    figure
    plot(1:N,ecg,'b'), hold on;
    plot(qrs_i_raw, qrs_amp_raw, 'b*');
    xlim([1 N]);    
end






