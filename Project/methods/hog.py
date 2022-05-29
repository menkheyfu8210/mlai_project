import numpy as np
import scipy.signal as sig
import os

def gradient(img, kernel='default'):
    if kernel == 'sobel':
        kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    elif kernel == 'prewitt':
        kernel_x = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    else:
        kernel_x = np.array([[-1, 0, 1]])
        kernel_y = np.array([[-1], [0], [1]])
    G_x = sig.convolve2d(img, kernel_x, mode='same') 
    G_y = sig.convolve2d(img, kernel_y, mode='same')
    return [G_x, G_y]
    
def nearestBins(bins, angle):
    bins = np.asarray(bins)
    return np.argsort(np.abs(bins-angle))[0:2]

def histogram(patchAngle, patchMagnitude, bins=9):
    bin = np.zeros(bins)
    angleStep = 180 // bins
    bins = np.arange(0, 180, angleStep)
    for i in range(8):
        for j in range(8):
            binIdx = nearestBins(bins, patchAngle[i,j])
            bin[binIdx[0]] += patchMagnitude[i,j] * (patchAngle[i,j] - bins[binIdx[1]]) / angleStep
            bin[binIdx[1]] += patchMagnitude[i,j] * (bins[binIdx[0]] - patchAngle[i,j]) / angleStep
    return bin

def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def hog(img, rescaleSize=(64,128), patchSize=8, bins=9, gradientKernel='default'):
    # Resize image to 64x128
    img = np.resize(img, rescaleSize)
    # Compute image gradients using convolution operators
    G = gradient(img, gradientKernel)
    # Calculate magnitude and orientation for each pixel
    M = np.sqrt(np.square(G[0]) + np.square(G[1]))
    a = np.arctan2(G[1], G[0])
    # Split both the magnitude and angle matrices in patches
    Mpatches = split(M, patchSize, patchSize)
    apatches = split(a, patchSize, patchSize)
    # Generate the histograms for the patches
    histograms = np.zeros((len(Mpatches), bins))
    for i in range(len(Mpatches)):
        histograms[i,:] = histogram(apatches[i], Mpatches[i], bins)
    # Normalize the gradients of bigger patches and build features vector
    patches = []
    for i in range(len(histograms) - patchSize - 1):
        if i % (patchSize) != 0 or i == 0:
            # Create the "histogram" for the bigger patch
            patch = np.concatenate((histograms[i,:], 
                                histograms[i + 1,:], 
                                histograms[i + patchSize,:], 
                                histograms[i + patchSize + 1,:]))
            # Apply normalization
            patch = patch / np.linalg.norm(patch)
            patches.append(patch)
    return np.hstack(patches)

    