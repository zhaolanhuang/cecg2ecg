from sklearn.decomposition import FastICA

def fastICA(x, n_components):
    ica = FastICA()
    S_ = ica.fit_transform(x)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix
    return S_, A_, ica