import cv2
import joblib
import numpy as np
import pandas as pd
import time

from methods.classifiers import SVM, KNN, ParzenWindows
from methods.detector import Detector
from methods.feature_extractor import extractFeatures, loadFeatures
from skimage.io import imread

def main(args = None):

    ###############
    # DEBUG UTILS #
    ###############
    preExtractedFeatures = True
    useSVMs = False
    preTrainedSVMs = True
    useKNN = False
    useParzenWindows = True

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
    # Load features and labels for testing
    loadFeatures(testFeatures, testLabels, True)
    
    #########################
    # CLASSIFICATION W/ SVM #
    #########################
    # Train (or load) a bunch of SVMs with different kernels and varying 
    # regularization params, then validate each one
    if useSVMs:
        C = [0.001, 0.01, 0.1, 1, 10]
        results = []
        for c in C:
            svm = SVM(C=c, kernel='linear', pretrained=preTrainedSVMs)
            if not preTrainedSVMs:
                svm.train(trainFeatures, trainLabels)
            results.extend(svm.validate(testFeatures, testLabels))
        for c in C:
            svm = SVM(C=c, kernel='poly', pretrained=preTrainedSVMs)
            if not preTrainedSVMs:
                svm.train(trainFeatures, trainLabels)
            results.extend(svm.validate(testFeatures, testLabels))
        for c in C:
            svm = SVM(C=c, kernel='sigmoid', pretrained=preTrainedSVMs)
            if not preTrainedSVMs:
                svm.train(trainFeatures, trainLabels)
            results.extend(svm.validate(testFeatures, testLabels))
        for c in C:
            svm = SVM(C=c, kernel='rbf', pretrained=preTrainedSVMs)
            if not preTrainedSVMs:
                svm.train(trainFeatures, trainLabels)
            results.extend(svm.validate(testFeatures, testLabels))
        results = np.reshape(np.array(results), (20, 7))
        df = pd.DataFrame(results, columns = ['kernel','C','accuracy', 'precision_np', 'precision_p', 'recall_np', 'recall_p'])
        joblib.dump(df, './validation_results/svm_testing_results.res')
        print(df)

    #########################
    # CLASSIFICATION W/ KNN #
    #########################
    if useKNN:
        # Initialize a bunch of KNN classifiers with varying K values and distance metrics
        K0 = int(np.sqrt(len(trainFeatures)))
        K = [K0 - 150, K0 - 100, K0 - 50, K0, K0 + 50, K0 + 100, K0 + 150]
        results = []
        knn = KNN(K0, 'cityblock')
        knn.train(trainFeatures, testFeatures)
        for k in K:
            knn.K = k
            print(f"cityblock K:{k}")
            results.extend(knn.validate(trainLabels, testLabels))
        knn = KNN(K0, 'euclidean')
        knn.train(trainFeatures, testFeatures)
        for k in K:
            knn.K = k
            print(f"euclidean K:{k}")
            results.extend(knn.validate(trainLabels, testLabels))
        knn = KNN(K0, 'minkowski')
        knn.train(trainFeatures, testFeatures)
        for k in K:
            knn.K = k
            print(f"minkowski K:{k}")
            results.extend(knn.validate(trainLabels, testLabels))
        joblib.dump(results, './validation_results/test.res')
        results = np.reshape(np.array(results), (21, 7))
        df = pd.DataFrame(results, columns = ['metric','K','accuracy', 'precision_np', 'precision_p', 'recall_np', 'recall_p'])
        joblib.dump(df, './validation_results/knn_testing_results.res')
        print(df)
    
    ####################################
    # CLASSIFICATION W/ PARZEN WINDOWS #
    ####################################
    if useParzenWindows:
        # Initialize a bunch of Parzen Windows classifiers with varying h values and kernel types
        H = [0.1, 0.25, 0.5, 0.75, 1]
        results = []
        for h in H:
            pw = ParzenWindows(h, 'rect', trainFeatures, testFeatures)
            print(f"rect h:{h}")
            results.extend(pw.validate(testFeatures, testLabels))
        for h in H:
            pw = ParzenWindows(h, 'tri', trainFeatures, testFeatures)
            print(f"tri h:{h}")
            results.extend(pw.validate(testFeatures, testLabels))
        for h in H:
            pw = ParzenWindows(h, 'gaussian', trainFeatures, testFeatures)
            print(f"gaussian h:{h}")
            results.extend(pw.validate(testFeatures, testLabels))
        for h in H:
            pw = ParzenWindows(h, 'dexp', trainFeatures, testFeatures)
            print(f"dexp h:{h}")
            results.extend(pw.validate(testFeatures, testLabels))
        joblib.dump(results, './validation_results/test.res')
        results = np.reshape(np.array(results), (20, 7))
        df = pd.DataFrame(results, columns = ['kernel','h','accuracy', 'precision_np', 'precision_p', 'recall_np', 'recall_p'])
        joblib.dump(df, './validation_results/knn_testing_results.res')
        print(df)
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