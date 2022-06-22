import cv2
import time

import joblib

from methods.classifiers import SVM, KNN, ParzenWindows
from methods.detector import Detector
from methods.feature_extractor import extractFeatures, loadFeatures
from skimage.io import imread

def main(args = None):

    ###############
    # DEBUG UTILS #
    ###############
    preExtractedFeatures = True
    preTrainedSVMs = True

    ######################
    # FEATURE EXTRACTION #
    ######################
    if not preExtractedFeatures:
        extractFeatures()
    trainFeatures = []
    trainLabels = []
    # Load features and labels for training
    loadFeatures(trainFeatures, trainLabels)
    testFeatures = []
    testLabels = []
    # Load features and labels for training
    loadFeatures(testFeatures, testLabels, True)

    #########################
    # CLASSIFICATION W/ SVM #
    #########################
    if not preTrainedSVMs:
        # Train a bunch of linear SVMs with varying regularization parameter
        C = [0.01, 0.1, 1, 10, 100]
        for c in C:
            svm = SVM(C=c, kernel='linear')
            svm.train(trainFeatures, trainLabels)
        # Train a bunch of poly SVMs with varying regularization parameter
        for c in C:
            svm = SVM(C=c, kernel='poly')
            svm.train(trainFeatures, trainLabels)
        # Train a bunch of sigmoid SVMs with varying regularization parameter
        for c in C:
            svm = SVM(C=c, kernel='sigmoid')
            svm.train(trainFeatures, trainLabels)
        # Train a bunch of rbf SVMs with varying regularization parameter
        for c in C:
            svm = SVM(C=c, kernel='rbf')
            svm.train(trainFeatures, trainLabels)
    
    svm = SVM(C=1, kernel='linear', pretrained=True)
    svm.validate(testFeatures, testLabels, _print=True)
    #########################
    # CLASSIFICATION W/ KNN #
    #########################
    # Initialize a bunch of KNN classifiers with varying K values
    #knn = KNN(10, 'euclidean', trainFeatures, trainLabels)
    #knn = KNN(10, 'euclidean', trainFeatures, trainLabels)
    #knn = KNN(10, 'euclidean', trainFeatures, trainLabels)
    #knn = KNN(10, 'euclidean', trainFeatures, trainLabels)
    #pw = ParzenWindows(0.2, 'gaussian', trainFeatures)

    """imgs = [imread('./test/test1.png', as_gray=True),
            imread('./test/test2.png', as_gray=True),
            imread('./test/test3.png', as_gray=True),
            imread('./test/test4.png', as_gray=True)]"""
            
    """for im in imgs:
        start_time = time.time()
        im = resize(im, (240, 320))

        detector = Detector(svm,
                        stepSize = [6,6],
                        downscale = 1.25,
                        nmsThreshold = 0.05)
        detections = detector.process(im)

        # Display the results after performing NMS
        for (x_tl, y_tl, s, w, h) in detections:

            # Draw the detections
            cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0,0,0))
            #cv2.putText(im, "%.2f" % s, (x_tl, y_tl - 10), cv2.FONT_HERSHEY_PLAIN, 1.1, (0,0,0), 2, cv2.LINE_AA)

        print(f"elapsed:{time.time() - start_time}")
        cv2.imshow("Detected pedestrians", im)
        cv2.waitKey()"""

if __name__ == "__main__":
    main()