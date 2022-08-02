import glob
import joblib
import os

from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
from tqdm import tqdm

SCALE_FACTOR = 64/18 # For rescaling images to 64x128 from 18x36

class FeatureExtractor():
    """Extraction of HOG features from images.
    """

    def __init__(self) -> None:
        # Create directories if needed
        if not os.path.exists('./Project/features/'):
            os.makedirs('./Project/features/')
        flag = True
        self.posTrainFeatPath = './Project/features/train/pos/'
        if not os.path.exists(self.posTrainFeatPath):
            os.makedirs(self.posTrainFeatPath)
            flag = False
        self.negTrainFeatPath = './Project/features/train/neg/'
        if not os.path.exists(self.negTrainFeatPath):
            os.makedirs(self.negTrainFeatPath)
            flag = False
        self.preExtractedTrainFeat = flag
        flag = True
        self.posTestFeatPath = './Project/features/test/pos/'
        if not os.path.exists(self.posTestFeatPath):
            os.makedirs(self.posTestFeatPath)
            flag = False
        self.negTestFeatPath = './Project/features/test/neg/'
        if not os.path.exists(self.negTestFeatPath):
            os.makedirs(self.negTestFeatPath)
            flag = False
        self.preExtractedTestFeat = flag
        # Paths to the training datasets
        base_1_neg_train = './Project/dataset/DC-ped-dataset_base/1/non-ped_examples/'
        base_2_neg_train = './Project/dataset/DC-ped-dataset_base/2/non-ped_examples/'
        base_3_neg_train = './Project/dataset/DC-ped-dataset_base/3/non-ped_examples/'
        base_1_pos_train = './Project/dataset/DC-ped-dataset_base/1/ped_examples/'
        base_2_pos_train = './Project/dataset/DC-ped-dataset_base/2/ped_examples/'
        base_3_pos_train = './Project/dataset/DC-ped-dataset_base/3/ped_examples/'
        # Paths to the test datasets
        base_t1_pos_test = './Project/dataset/DC-ped-dataset_base/T1/ped_examples/'
        base_t1_neg_test = './Project/dataset/DC-ped-dataset_base/T1/non-ped_examples/'
        base_t2_pos_test = './Project/dataset/DC-ped-dataset_base/T2/ped_examples/'
        base_t2_neg_test = './Project/dataset/DC-ped-dataset_base/T2/non-ped_examples/'
        # Group together the paths
        self.posTrainPaths = [base_1_pos_train, base_2_pos_train, base_3_pos_train]
        self.negTrainPaths = [base_1_neg_train, base_2_neg_train, base_3_neg_train]
        self.posTestPaths = [base_t1_pos_test, base_t2_pos_test]
        self.negTestPaths = [base_t1_neg_test, base_t2_neg_test]
        # Check if the dataset is there
        for path in self.posTrainPaths:
            if not os.path.exists(path):
                raise FileNotFoundError("Can't extract features from " + path + ": no such directory.")
        for path in self.negTrainPaths:
            if not os.path.exists(path):
                raise FileNotFoundError("Can't extract features from " + path + ": no such directory.")
        for path in self.posTestPaths:
            if not os.path.exists(path):
                raise FileNotFoundError("Can't extract features from " + path + ": no such directory.")
        for path in self.negTestPaths:
            if not os.path.exists(path):
                raise FileNotFoundError("Can't extract features from " + path + ": no such directory.")

    def extract_training_features(self):
        """Extracts the Histogram Of Gradients feature vectors from the training dataset.
        """
        if not self.preExtractedTrainFeat:
            print('Starting training feature extraction. This will take a while.')
            for (i, path) in enumerate(self.posTrainPaths):
                # Set up a progress bar on the enumerable
                pbar = tqdm(glob.glob(os.path.join(path, '*')))
                for sample in pbar:
                    pbar.set_description('Extracting positive training features (dataset ' + str(i + 1) + ')')
                    # Load the image and rescale it to 64x128
                    img = imread(sample)
                    img = rescale(img, SCALE_FACTOR, anti_aliasing=False)
                    # Extract the hog feature from the image
                    feature = hog(img, cells_per_block=(2,2))
                    fileName = str(i + 1) + os.path.split(sample)[1].split(".")[0] + ".feat"
                    joblib.dump(feature, os.path.join(self.posTrainFeatPath, fileName))
            for (i, path) in enumerate(self.negTrainPaths):
                # Set up a progress bar on the enumerable
                pbar = tqdm(glob.glob(os.path.join(path, '*')))
                for sample in pbar:
                    pbar.set_description('Extracting negative training features (dataset ' + str(i + 1) + ')')
                    img = imread(sample)
                    img = rescale(img, SCALE_FACTOR, anti_aliasing=False)
                    # Extract the hog feature from the image
                    feature = hog(img, cells_per_block=(2,2))
                    fileName = str(i + 1) + os.path.split(sample)[1].split(".")[0] + ".feat"
                    joblib.dump(feature, os.path.join(self.negTrainFeatPath, fileName))
        return self.load_train_features()
        
    def extract_testing_features(self):
        """Extracts the Histogram Of Gradients feature vectors from the testing dataset.
        """
        if not self.preExtractedTestFeat:
            print('Starting testing feature extraction. This will take a while.')
            for (i, path) in enumerate(self.posTestPaths):
                # Set up a progress bar on the enumerable
                pbar = tqdm(glob.glob(os.path.join(path, '*')))
                for sample in pbar:
                    pbar.set_description('Extracting positive testing features (dataset T' + str(i + 1) + ')')
                    img = imread(sample)
                    img = rescale(img, SCALE_FACTOR, anti_aliasing=False)
                    # Extract the hog feature from the image
                    feature = hog(img, cells_per_block=(2,2))
                    fileName = str(i + 1) + os.path.split(sample)[1].split(".")[0] + ".feat"
                    joblib.dump(feature, os.path.join(self.posTestFeatPath, fileName))
            for (i, path) in enumerate(self.negTestPaths):
                # Set up a progress bar on the enumerable
                pbar = tqdm(glob.glob(os.path.join(path, '*')))
                for sample in pbar:
                    pbar.set_description('Extracting negative testing features (dataset T' + str(i + 1) + ')')
                    img = imread(sample)
                    img = rescale(img, SCALE_FACTOR, anti_aliasing=False)
                    # Extract the hog feature from the image
                    feature = hog(img, cells_per_block=(2,2))
                    fileName = str(i + 1) + os.path.split(sample)[1].split(".")[0] + ".feat"
                    joblib.dump(feature, os.path.join(self.negTestFeatPath, fileName))
        return self.load_test_features()

    def load_train_features(self):
        """Loads pre-extracted training feature vectors.
        """
        print('Starting to load training features. This will take a while.')
        features = []
        labels = []
        # Set up a progress bar on the enumerable
        pbar = tqdm(glob.glob(os.path.join(self.posTrainFeatPath, '*')))
        # Load the positive features
        for path in pbar:
            pbar.set_description('Loading positive training features')
            f = joblib.load(path)
            features.append(f)
            labels.append(1)
        # Set up a progress bar on the enumerable
        pbar = tqdm(glob.glob(os.path.join(self.negTrainFeatPath, '*')))
        # Load the negative features
        for path in pbar:
            pbar.set_description('Loading negative training features')
            f = joblib.load(path)
            features.append(f)
            labels.append(0)
        return (features, labels)

    def load_test_features(self):
        """Loads pre-extracted testing feature vectors.
        """
        print('Starting to load testing features. This will take a while.')
        features = []
        labels = []
        # Set up a progress bar on the enumerable
        pbar = tqdm(glob.glob(os.path.join(self.posTestFeatPath, '*')))
        # Load the positive features
        for path in pbar:
            pbar.set_description('Loading positive testing features')
            f = joblib.load(path)
            features.append(f)
            labels.append(1)
        # Set up a progress bar on the enumerable
        pbar = tqdm(glob.glob(os.path.join(self.negTestFeatPath, '*')))
        # Load the negative features
        for path in pbar:
            pbar.set_description('Loading negative testing features')
            f = joblib.load(path)
            features.append(f)
            labels.append(0)
        return (features, labels)