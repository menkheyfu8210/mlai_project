import numpy as np
from sklearn.decomposition import IncrementalPCA

def normalize(V):
    # Normalize matrix rows
    for i in range(V.shape[0]):
        V[i,:] = V[i,:] if np.linalg.norm(V[i,:]) == 0 else V[i,:] / np.linalg.norm(V[i,:])
    return V

def pca(x):
    # Normalize the input matrix rows
    x = normalize(x)
    # Centre the data around the mean
    m = np.mean(x, axis=1)
    Xc = x - m[:,np.newaxis]
    # Compute the covariance matrix
    C = np.cov(Xc, rowvar=False)
    # Extract eigenvectors and eigenvalues of the covariance matrix
    lambdas, U = np.linalg.eigh(C)
    # Oorder the eigenvalues from largest to smallest
    best_eig_idxs = np.argsort(lambdas)[::-1]
    best_eig = lambdas[best_eig_idxs]
    best_U = U[:,best_eig_idxs]

    # Keep only as many eigenvectors as to keep 80% of the data variances
    y = np.cumsum(best_eig)/np.sum(best_eig)
    N = np.where(y >= 0.8)
    print(len(y))
    print(N[0].shape)
