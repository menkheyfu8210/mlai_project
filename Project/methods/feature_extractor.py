import glob
import joblib
import os

from skimage.feature import hog
from skimage.io import imread


def extractFeatures():
    """Extracts the Histogram Of Gradients feature vectors from the dataset.
    """
    # Paths to the training datasets
    base_1_neg_train = './../Dataset/DC-ped-dataset_base/1/non-ped_examples/'
    base_2_neg_train = './../Dataset/DC-ped-dataset_base/2/non-ped_examples/'
    base_3_neg_train = './../Dataset/DC-ped-dataset_base/3/non-ped_examples/'
    base_1_pos_train = './../Dataset/DC-ped-dataset_base/1/ped_examples/'
    base_2_pos_train = './../Dataset/DC-ped-dataset_base/2/ped_examples/'
    base_3_pos_train = './../Dataset/DC-ped-dataset_base/3/ped_examples/'
    # Paths to the test datasets
    base_t1_pos_test = './../Dataset/DC-ped-dataset_base/T1/ped_examples/'
    base_t1_neg_test = './../Dataset/DC-ped-dataset_base/T1/non-ped_examples/'
    base_t2_pos_test = './../Dataset/DC-ped-dataset_base/T2/ped_examples/'
    base_t2_neg_test = './../Dataset/DC-ped-dataset_base/T2/non-ped_examples/'
    # Group together the paths
    posTrainPaths = [base_1_pos_train, base_2_pos_train, base_3_pos_train]
    negTrainPaths = [base_1_neg_train, base_2_neg_train, base_3_neg_train]
    posTestPaths = [base_t1_pos_test, base_t2_pos_test]
    negTestPaths = [base_t1_neg_test, base_t2_neg_test]

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

    for path in posTrainPaths:
        samples = glob.glob(os.path.join(path, '*'))
        for sample in samples:
            img = imread(sample)
            feature = hog(img, cells_per_block=(2,2))
            fileName = os.path.split(sample)[1].split(".")[0] + ".feat"
            joblib.dump(feature, os.path.join(posTrainFeatPath, fileName))
    print("Extracted positive training features.")
    for path in negTrainPaths:
        samples = glob.glob(os.path.join(path, '*'))
        for sample in samples:
            img = imread(sample)
            feature = hog(img, cells_per_block=(2,2))
            fileName = os.path.split(sample)[1].split(".")[0] + ".feat"
            joblib.dump(feature, os.path.join(negTrainFeatPath, fileName))
    print("Extracted negative training features.")
    for path in posTestPaths:
        samples = glob.glob(os.path.join(path, '*'))
        for sample in samples:
            img = imread(sample)
            feature = hog(img, cells_per_block=(2,2))
            fileName = os.path.split(sample)[1].split(".")[0] + ".feat"
            joblib.dump(feature, os.path.join(posTestFeatPath, fileName))
    print("Extracted positive testing features.")
    for path in negTestPaths:
        samples = glob.glob(os.path.join(path, '*'))
        for sample in samples:
            img = imread(sample)
            feature = hog(img, cells_per_block=(2,2))
            fileName = os.path.split(sample)[1].split(".")[0] + ".feat"
            joblib.dump(feature, os.path.join(negTestFeatPath, fileName))
    print("Extracted negative testing features.")

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