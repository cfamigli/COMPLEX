
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def do_PCA(X, n_components):
    # perform PCA on matrix X (rows=num_ensembles x cols=num_parameters=n_components).
    # returns number of principal components needed for >95% variance explained.
    # (alternatively can edit to return the components themselves, etc)

    if (X==X[0]).all(): # if there is only one unique ensemble member
        reduced_dimensionality = float('nan')
    else:
        X_scaled = StandardScaler().fit_transform(X) # standardize features by removing the mean and scaling to unit variance
        X_scaled = X_scaled[~np.isnan(X_scaled).any(axis=1)] # remove any row that has NaN
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        reduced_dimensionality = np.argwhere(np.cumsum(pca.explained_variance_ratio_)>0.95)[0][0]
    return reduced_dimensionality
