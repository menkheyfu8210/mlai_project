import os
import random
import joblib
import numpy as np
import matplotlib.pyplot as plt
import time

from utils import loader
from methods.hog import hog
from methods.svm import svm, getMetrics
from PIL import Image

def main(args = None):

    # To keep random number generation consistent between runs
    random.seed(0)

    # These variables set respectively the number of positive and negative training samples
    N = 5000
    P = 8000
    # Load the training images
    #posData = loader.loadImgs('../Dataset/Pedestrians/', P)
    #negData = loader.loadImgs('../Dataset/NonPedestrians/', N)
    # Extract the features from the training images using Histogram of Gradients
    #posFeatures = []
    #for p in posData:
    #    posFeatures.append(hog(p))
    #posFeatures = np.hstack(posFeatures)
    #posFeatures = np.reshape(posFeatures, (P, int(posFeatures.shape[0] / P)))
    #posFeatures = np.nan_to_num(posFeatures)
    #np.save(os.path.join('./data', 'posFeatures'), posFeatures)
    #negFeatures = []
    #for n in negData:
    #    negFeatures.append(hog(n))
    #negFeatures = np.hstack(negFeatures)
    #negFeatures = np.reshape(negFeatures, (N, int(negFeatures.shape[0] / N)))
    #negFeatures = np.nan_to_num(negFeatures)
    #np.save(os.path.join('./data', 'negFeatures'), negFeatures)

    # Save time, load the features computed with the above method
    #posFeatures = np.load('data/posFeatures.npy')
    #negFeatures = np.load('data/negFeatures.npy')

    # Collect training data
    #trainData = np.vstack((posFeatures,negFeatures))
    # Generate corresponding labels, 0 for positive and 1 for negative
    #P = posFeatures.shape[0]
    #N = negFeatures.shape[0]
    #trainLabels = np.hstack((np.zeros((P)), np.ones((N))))
    
    # Generate an SVM on the training data
    #model = svm(trainData, trainLabels)

    # Save time, load the pre trained model
    model = joblib.load(os.path.join('./data', 'trained_linear_svm.sav'))
    
    #testData = loader.loadImgs('../Dataset/test/')
    #testLabels = np.hstack((np.zeros((45)), np.ones((74))))
    # Extract the features from the test images using hog
    #features = []
    #for t in testData:
    #    features.append(hog(t))
    #features = np.hstack(features)
    #features = np.reshape(features, (len(testData), int(features.shape[0] / len(testData))))
    #features = np.nan_to_num(features)
    # Classify the testing set and build the confusion matrix
    #[cfs, accuracy, precision_0, precision_1, recall_0, recall_1] = getMetrics(model, features, testLabels)
    #print('Classifier accuracy: ' + "{0:.2f}".format(accuracy*100) + '%')
    #print('Precision w.r.t class 0: ' + "{0:.2f}".format(precision_0))
    #print('Precision w.r.t class 1: ' + "{0:.2f}".format(precision_1))
    #print('Recall w.r.t class 0: ' + "{0:.2f}".format(recall_0))
    #print('Recall w.r.t class 1: ' + "{0:.2f}".format(recall_1))

if __name__ == "__main__":
    main()