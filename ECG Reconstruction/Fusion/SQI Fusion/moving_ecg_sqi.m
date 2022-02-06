function [iSQIs, vSQIs] = moving_ecg_sqi(ecg, fs)
N = length(ecg);
window = round(fs * 1); %1 sec
iSQIs = 1:window:N-window;
vSQIs = [];

for i = iSQIs
    snippet = ecg(i:i+window);
    [basSQI,pliSQI,qrsSQI] = calcSQI(snippet,fs);
    
    vSQIs(end+1) = qrsSQI;
    
end

