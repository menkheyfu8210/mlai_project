import glob
import joblib
import os

from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SCALE_FACTOR = 64/18 # For rescaling images to 64x128 from 18x36

class FeatureExtractor():
    """Extraction of HOG features from images.
    """

    def __init__(self) -> None:
        # Create directories if needed
        self.featPath = './Project/features/'
        if not os.path.exists(self.featPath ):
            os.makedirs(self.featPath)
        flag = True
        self.trainFeatPath = './Project/features/train/'
        if not os.path.exists(self.trainFeatPath):
            os.makedirs(self.trainFeatPath)
            flag = False
        self.preExtractedTrainFeat = flag
        flag = True
        self.testFeatPath = './Project/features/test/'
        if not os.path.exists(self.testFeatPath):
            os.makedirs(self.testFeatPath)
            flag = False
        self.preExtractedTestFeat = flag
        flag = True
        self.addFeatPath = './Project/features/add/'
        if not os.path.exists(self.addFeatPath):
            os.makedirs(self.addFeatPath)
            flag = False
        self.preExtractedAddFeat = flag
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
        # Paths to the additional datasets
        add_1 = './Project/dataset/DC-ped-dataset_add-1/1/add_non-ped_images'
        add_2 = './Project/dataset/DC-ped-dataset_add-2/2/add_non-ped_images'
        add_3 = './Project/dataset/DC-ped-dataset_add-3/3/add_non-ped_images'
        # Group together the paths
        self.posTrainPaths = [base_1_pos_train, base_2_pos_train, base_3_pos_train]
        self.negTrainPaths = [base_1_neg_train, base_2_neg_train, base_3_neg_train]
        self.posTestPaths = [base_t1_pos_test, base_t2_pos_test]
        self.negTestPaths = [base_t1_neg_test, base_t2_neg_test]
        self.addPaths = [add_1, add_2, add_3]
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
        for path in self.addPaths:
            if not os.path.exists(path):
                raise FileNotFoundError("Can't extract features from " + path + ": no such directory.")

    def extract_training_features(self):
        """Extracts the Histogram Of Gradients feature vectors from the training 
        dataset, and also creates a list of labels.
        """
        if not self.preExtractedTrainFeat:
            features = []
            labels = []
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
                    features.append(hog(img, cells_per_block=(2,2)))
                    labels.append(1)
            for (i, path) in enumerate(self.negTrainPaths):
                # Set up a progress bar on the enumerable
                pbar = tqdm(glob.glob(os.path.join(path, '*')))
                for sample in pbar:
                    pbar.set_description('Extracting negative training features (dataset ' + str(i + 1) + ')')
                    img = imread(sample)
                    img = rescale(img, SCALE_FACTOR, anti_aliasing=False)
                    # Extract the hog feature from the image
                    features.append(hog(img, cells_per_block=(2,2)))
                    labels.append(0)
            # Apply scaling and PCA to the features
            scaler = StandardScaler()
            scaler.fit(features)
            features = scaler.transform(features)
            pca = PCA(n_components=0.9)
            pca.fit(features)
            features = pca.transform(features)
            # Save the features and labels
            joblib.dump(features, os.path.join(self.trainFeatPath, 'training_features.feat'))
            joblib.dump(labels, os.path.join(self.trainFeatPath, 'training_labels.feat'))
            # Save the fitted scaler and pca, we will need them for the testing data
            joblib.dump(scaler, os.path.join(self.featPath, 'scaler.mod'))
            joblib.dump(pca, os.path.join(self.featPath, 'pca.mod'))
        return self.load_train_features()
        
    def extract_testing_features(self):
        """Extracts the Histogram Of Gradients feature vectors from the testing dataset.
        """
        if not self.preExtractedTestFeat:
            features = []
            labels = []
            print('Starting testing feature extraction. This will take a while.')
            for (i, path) in enumerate(self.posTestPaths):
                # Set up a progress bar on the enumerable
                pbar = tqdm(glob.glob(os.path.join(path, '*')))
                for sample in pbar:
                    pbar.set_description('Extracting positive testing features (dataset T' + str(i + 1) + ')')
                    img = imread(sample)
                    img = rescale(img, SCALE_FACTOR, anti_aliasing=False)
                    # Extract the hog feature from the image
                    features.append(hog(img, cells_per_block=(2,2)))
                    labels.append(1)
            for (i, path) in enumerate(self.negTestPaths):
                # Set up a progress bar on the enumerable
                pbar = tqdm(glob.glob(os.path.join(path, '*')))
                for sample in pbar:
                    pbar.set_description('Extracting negative testing features (dataset T' + str(i + 1) + ')')
                    img = imread(sample)
                    img = rescale(img, SCALE_FACTOR, anti_aliasing=False)
                    # Extract the hog feature from the image
                    features.append(hog(img, cells_per_block=(2,2)))
                    labels.append(0)
            # Apply scaling and PCA to the features
            scaler = joblib.load(self.featPath + 'scaler.mod')
            features = scaler.transform(features)
            pca = joblib.load(self.featPath + 'pca.mod')
            features = pca.transform(features)
            # Save the features and labels
            joblib.dump(features, os.path.join(self.testFeatPath, 'testing_features.feat'))
            joblib.dump(labels, os.path.join(self.testFeatPath, 'testing_labels.feat'))
        return self.load_test_features()

    def extract_add_features(self):
        """Extracts the Histogram Of Gradients feature vectors from the additional dataset.
        """
        if not self.preExtractedAddFeat:
            features = []
            print('Starting additional feature extraction. This will take a while.')
            for (i, path) in enumerate(self.addTestPaths):
                # Set up a progress bar on the enumerable
                pbar = tqdm(glob.glob(os.path.join(path, '*')))
                for sample in pbar:
                    pbar.set_description('Extracting additional features (dataset ' + str(i + 1) + ')')
                    img = imread(sample)
                    img = rescale(img, SCALE_FACTOR, anti_aliasing=False)
                    # Extract the hog feature from the image
                    features.append(hog(img, cells_per_block=(2,2)))
            # Apply scaling and PCA to the features
            scaler = joblib.load(self.featPath + 'scaler.mod')
            features = scaler.transform(features)
            pca = joblib.load(self.featPath + 'pca.mod')
            features = pca.transform(features)
            # Save the features and labels
            joblib.dump(features, os.path.join(self.addFeatPath, 'add_features.feat'))
        return self.load_add_features()

    def load_train_features(self):
        """Loads pre-extracted training feature vectors.
        """
        print('Loading training features... ', end='')
        features = joblib.load(os.path.join(self.trainFeatPath, 'training_features.feat'))
        labels = joblib.load(os.path.join(self.trainFeatPath, 'training_labels.feat'))
        print('Done.')
        return (features, labels)

    def load_test_features(self):
        """Loads pre-extracted testing feature vectors.
        """
        print('Loading testing features... ', end='')
        features = joblib.load(os.path.join(self.testFeatPath, 'testing_features.feat'))
        labels = joblib.load(os.path.join(self.testFeatPath, 'testing_labels.feat'))
        print('Done.')
        return (features, labels)

    def load_add_features(self):
        """Loads pre-extracted additional feature vectors.
        """
        print('Loading additional features... ', end='')
        features = joblib.load(os.path.join(self.addFeatPath, 'add_features.feat'))
        labels = joblib.load(os.path.join(self.addFeatPath, 'add_labels.feat'))
        print('Done.')
        return (features, labels)