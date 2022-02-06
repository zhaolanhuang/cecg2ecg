function [basSQI,pliSQI,qrsSQI] = calcSQI(ecg,fs, window)
if nargin < 3
    window = [];
end

N = length(ecg);
% Sum 0-1Hz psd
[pxx, f] = periodogram(ecg, window, N,fs);

pxx_0_1 = pxx(find(0<=f & f<=1));
pxx_0_40 = pxx(find(0<=f & f<=40));
basSQI = 1 - sum(pxx_0_1) / (1e-6 + sum(pxx_0_40));

pxx_49_51 = pxx(find(49<=f & f<=51));
pxx_40_60 = pxx(find(40<=f & f<=60));
pliSQI = 1 - sum(pxx_49_51) / (1e-6 + sum(pxx_40_60));

pxx_5_15 = pxx(find(5<=f & f<=15));
pxx_5_40 = pxx(find(5<=f & f<=40));
qrsSQI = sum(pxx_5_15) / (1e-6 + sum(pxx_5_40));
end

