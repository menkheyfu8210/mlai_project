import glob
import joblib
import os

from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize

def extractFeatures():
    """Extracts the Histogram Of Gradients feature vectors from the dataset.
    """
    # Paths to the dataset
    posPath = './../Dataset/Pedestrians/'
    negPath = './../Dataset/NonPedestrianPatches/'
    # Create directories if needed
    if not os.path.exists('./features/'):
        os.makedirs('./features/')
    posTrainFeatPath = './features/train/pos/'
    if not os.path.exists(posTrainFeatPath):
        os.makedirs(posTrainFeatPath)
    posTestFeatPath = './features/test/pos/'
    if not os.path.exists(posTestFeatPath):
        os.makedirs(posTestFeatPath)
    negTrainFeatPath = './features/train/neg/'
    if not os.path.exists(negTrainFeatPath):
        os.makedirs(negTrainFeatPath)
    negTestFeatPath = './features/test/neg/'
    if not os.path.exists(negTestFeatPath):
        os.makedirs(negTestFeatPath)
    # Extraction of the positive samples features
    posSamples = glob.glob(os.path.join(posPath, '*'))
    # Cutoff for training/test split
    P = int(len(posSamples) * 0.8)
    i = 0
    for imgPath in posSamples:
        im = imread(imgPath)
        im = resize(im, (128, 64), anti_aliasing=True)
        f = hog(im, cells_per_block=(2,2))
        fn = os.path.split(imgPath)[1].split(".")[0] + ".feat"
        if i < P:
            path = os.path.join(posTrainFeatPath, fn)
        else:
            path = os.path.join(posTestFeatPath, fn)
        i = i + 1
        joblib.dump(f, path)
    # Extraction of the negative samples features
    negSamples = glob.glob(os.path.join(negPath, '*'))
    # Cutoff for training/test split
    N = int(len(negSamples) * 0.8)
    i = 0
    for imgPath in negSamples:
        im = imread(imgPath)
        f = hog(im, cells_per_block=(2,2))
        fn = os.path.split(imgPath)[1].split(".")[0] + ".feat"
        if i < N:
            path = os.path.join(negTrainFeatPath, fn)
        else:
            path = os.path.join(negTestFeatPath, fn)
        i = i + 1
        joblib.dump(f, path)

def loadFeatures(features, labels, test=False):
    """Loads pre-extracted feature vectors.

    Parameters
    ----------
    features : list
        Empty list to be filled with feature vectors.

    labels : list
        Empty list to be filled with the class labels associated to the feature 
        vectors.

    test : bool
        True to load the testing features, False to load the training features.
    """
    if not test:
        posFeatPath = './features/train/pos/'
        negFeatPath = './features/train/neg/'
    else:
        posFeatPath = './features/test/pos/'
        negFeatPath = './features/test/neg/'
    # Load the positive features
    for path in glob.glob(os.path.join(posFeatPath, '*')):
        f = joblib.load(path)
        features.append(f)
        labels.append(1)
    # Load the negative features
    for path in glob.glob(os.path.join(negFeatPath, '*')):
        f = joblib.load(path)
        features.append(f)
        labels.append(0)