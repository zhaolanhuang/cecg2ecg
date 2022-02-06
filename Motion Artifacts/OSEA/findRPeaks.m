function [peaks, typeCode] = findRPeaks(ecgSignal, amp, fs)
%findRPeaks - Finds the peaks in an ECG signal using OSEA
%
%
% Syntax:  peaks = findRPeaks(ecgSignal, amp, fs)
%
% Inputs:
%    ecgSignal - ECG signal
%    amp - 		OSEA requires a minimum amplitude to work properly. Hence, an
%				amplification factor is included.
%    fs -       OSEA requires a sampling rate of 200 Hz. Hence, it will be
%               resamples if required
%
% Outputs:
%    peaks - Vector with the detected peak locations in samples 
%	 typeCode - Vector with the type code of the detected peaks
%
% Example: 
%	 load('meditPublicSigs.mat')
%    peaks = findPeaksInRecord(meditPublicSigs{1}.channels{1}.data, 500, meditPublicSigs{1}.channels{1}.fs)
%
% Other m-files required: OSEA.mex 
% Subfunctions: none
% MAT-files required: none
%
% See also: OSEA.m 

% Author: Tobias Wartzek, Dr.-Ing., Chair of Medical Information Technology
% RWTH Aachen University
% email address: wartzek@hia.rwth-aachen.de  
% Website: http://www.medit.hia.rwth-aachen.de
% July 2014
% Version 1.0
%------------- BEGIN CODE --------------

ecgSignal = double(ecgSignal(:)');

%% OSEA requires the signal to be sampled at 200 Hz
	if fs~=200
		signal_200Hz = resample(ecgSignal, 200, double(fs));
	else
		signal_200Hz = ecgSignal;
	end

%% OSEA does not find the first R-peaks due to initialization 
	% hence, extend the signal at the beginning for 10 seconds.
	sampExtend = 10 * 200;  % 10 seconds
	if length(signal_200Hz) > sampExtend
		signal4OSEA = [fliplr(signal_200Hz(1:sampExtend)) signal_200Hz];
	else
		sampExtend = length(signal_200Hz);
		signal4OSEA = [fliplr(signal_200Hz) signal_200Hz];
	end
    
%%	Use OSEA to find R-peaks
	skeleton = OSEA(signal4OSEA*amp);
	peaksEstim = find(skeleton > 0);
	typeCode = skeleton(peaksEstim);
	
	% correct for the signal extension
	typeCode(peaksEstim<sampExtend) = [];
	peaksEstim(peaksEstim<sampExtend) = [];
	peaksEstim = peaksEstim - sampExtend;
	
%% Find exact position in original signal
	dT=round(0.04*fs); % window around each peak to find maximum
	peaksEstim = peaksEstim * (fs/200);
	k=1;
	peaks = NaN * peaksEstim;
	for l = 1:length(peaksEstim)
		search_interval_start = round(max(1, peaksEstim(l)-dT));
		search_interval_end = round(min(length(ecgSignal), peaksEstim(l)+dT));
		[~,index] = max(ecgSignal(search_interval_start:search_interval_end));
		peak_pos = search_interval_start - 1 + index;
		if  k == 1 || peak_pos ~= peaks(k-1)
			peaks(k) = peak_pos;
			k = k+1;
		end
	end
    
    peaks=peaks(:);
	typeCode =  typeCode(:);
	typeCode(isnan(peaks)) = [];
    peaks(isnan(peaks)) = [];
