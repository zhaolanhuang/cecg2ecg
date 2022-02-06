% Implementation of Herleiksons algorithm
% max_rise_time [s]: maximum duration between two consecutive extrema points with opposite polarities
%                for conventonal ECG set to 3 ms, for capacitive ECG set to 4 ms.
% threshold_window [s]: default 64 ms.
% k: Threshold factor

function [vals,locs, ecg_diff] = find_pacespikes_herleikson(ecg, fs, max_rise_time, k, threshold_window)
N_threshold_window = round(fs * threshold_window);
N_max_rise_time = round(fs * max_rise_time);

% Differentiator y[n] = x[n] + x[n-1] - x[n-2] - x[n-3]
x = filter([1 1 -1 -1],[1 0 0 0],ecg);

x = x(:);
ecg_diff = x;
patent = 1;
if patent == 1
    [vals,locs] = herleikson_patent(x, N_max_rise_time, k, N_threshold_window);
    return;
end

[~, pklocs_positive] = findpeaks(x);
[~, pklocs_negative] = findpeaks(-x);
pklocs = sort([pklocs_positive;pklocs_negative]);
pks = x(pklocs);
n_peaks = length(pklocs);
vals = [];
locs = [];
if n_peaks < 2
    return
end

for i = 2:n_peaks
    if pklocs(i-1) < N_threshold_window
        continue
    end
    if pks(i) * pks(i-1) > 0
        continue;
    end
    if pklocs(i) - pklocs(i-1) > N_max_rise_time
        continue;
    end
    if ismember(pklocs(i-1), locs)
        continue;
    end
    x_threshold_window = x( max(1, pklocs(i-1)- N_threshold_window): max(1,pklocs(i-1) - 1));
    T = k * max(abs(x_threshold_window));
    if abs(pks(i)) >= T && abs(pks(i - 1)) >= T
        vals = [vals pks(i)];
        locs = [locs pklocs(i)];
    end
end

end

% US Patent US005682902A
% c: Threshold factor k
function [vals,locs] = herleikson_patent(x, max_rise_time, c, threshold_window)
N = length(x);
vals = [];
locs = [];
i = threshold_window + 1 ;
while i <= N-max_rise_time
    T = c * max(abs(x( max(1,i-threshold_window-max_rise_time):max(1,i-1-max_rise_time))));
    bPace = false;
    for j = i:i+max_rise_time
        if abs(x(j)) >= T
            for k = j+1:i+max_rise_time
                if x(j)*x(k) < 0 && abs(x(k)) >= T 
                    vals = [vals x(j)];
                    locs = [locs j];
                    bPace = true;
                    break;
                end
            end
        end
        if bPace
            break;
        end
    end
    if bPace
        i = k + 1;
    else
        i = i + 1;
    end

end
end

