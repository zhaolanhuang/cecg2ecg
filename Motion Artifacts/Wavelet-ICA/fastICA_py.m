% est_source: estimated Source signal (n_samples, n_source)
% A: estimated mixing matrix
% ica_obj: inner object of scikit-learn, needed for inverse Transformation
% x : array-like of shape (n_samples, n_features)
function [est_source, A, ica_obj] = fastICA_py(x, n_components)
fullpath = mfilename('fullpath'); 
[path,~] = fileparts(fullpath);
insert(py.sys.path,int32(0),path);
mod = py.importlib.import_module('fastICA');
out = mod.fastICA(x, n_components);
est_source = double(out{1});
A = double(out{2});
ica_obj = out{3};
end

