% Generate Data set for training the Parameter estimator for dynamic modell
% Workflow: Extraction cardiac cycles from UnoVis Auto2012 -> Exclude outliers of extracted cycles -> resample
%			-> Estimate parameters [a_i, theta_i, b_i] of rECG

run('extraction.m')
run('data_cleaning.m')
run('cyc_resample.m')
run('curve_fit.m')

disp("Finished!")