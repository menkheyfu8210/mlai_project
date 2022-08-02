import joblib
import numpy as np
import pandas as pd

from methods.classifiers import SVM, KNN, NaiveBayes
from methods.feature_extractor import FeatureExtractor

def main(args = None):

    ############
    # SETTINGS #
    ############
    useNB = True
    preTrainedNB = False
    useKNN = False
    preTrainedKNNs = False
    useSVMs = False
    preTrainedSVMs = False

    ##################################
    # FEATURE EXTRACTION AND LOADING #
    ##################################
    fe = FeatureExtractor()
    (trainFeatures, trainLabels) = fe.extract_training_features()
    (testFeatures, testLabels) = fe.extract_testing_features()
    
    #################################
    # CLASSIFICATION W/ NAIVE BAYES #
    #################################
    if useNB:
        results = []
        nb = NaiveBayes(pretrained=preTrainedNB)
        if not preTrainedNB:
            nb.train(trainFeatures, trainLabels)
        results.extend(nb.validate(nb.predict(testFeatures), testLabels))
        results = np.reshape(np.array(results), (1, 8))
        df = pd.DataFrame(results, columns = ['-','-','accuracy', 'precision_np', 'precision_p', 'recall_np', 'recall_p', 'f1'])
        joblib.dump(df, './Project/validation_results/nb_testing_results.res')
        print(df)

    #########################
    # CLASSIFICATION W/ KNN #
    #########################
    if useKNN:
        # Initialize a bunch of KNN classifiers with varying K values and distance metrics
        K = [k for k in range(1, 9000, 100)]
        results = []
        for k in K:
            knn = KNN(k, 'cityblock', pretrained=preTrainedKNNs)
            if not preTrainedKNNs:
                knn.train(trainFeatures, trainLabels)
            results.extend(knn.validate(knn.predict(testFeatures), testLabels))
        for k in K:
            knn = KNN(k, 'euclidean', pretrained=preTrainedKNNs)
            if not preTrainedKNNs:
                knn.train(trainFeatures, trainLabels)
            results.extend(knn.validate(knn.predict(testFeatures), testLabels))
        for k in K:
            knn = KNN(k, 'minkowski', pretrained=preTrainedKNNs)
            if not preTrainedKNNs:
                knn.train(trainFeatures, trainLabels)
            results.extend(knn.validate(knn.predict(testFeatures), testLabels))
        results = np.reshape(np.array(results), (270, 8))
        df = pd.DataFrame(results, columns = ['metric','K','accuracy', 'precision_np', 'precision_p', 'recall_np', 'recall_p', 'f1'])
        joblib.dump(df, './Project/validation_results/knn_testing_results.res')
        print(df)
        
    #########################
    # CLASSIFICATION W/ SVM #
    #########################
    # Train (or load) a bunch of SVMs with different kernels and varying 
    # regularization params, then validate each one
    if useSVMs:
        C = [0.01, 0.1, 1, 10, 100]
        results = []
        for c in C:
            svm = SVM(C=c, kernel='linear', pretrained=preTrainedSVMs)
            if not preTrainedSVMs:
                svm.train(trainFeatures, trainLabels)
            results.extend(svm.validate(svm.predict(testFeatures), testLabels))
        for c in C:
            svm = SVM(C=c, kernel='poly', pretrained=preTrainedSVMs)
            if not preTrainedSVMs:
                svm.train(trainFeatures, trainLabels)
            results.extend(svm.validate(svm.predict(testFeatures), testLabels))
        for c in C:
            svm = SVM(C=c, kernel='sigmoid', pretrained=preTrainedSVMs)
            if not preTrainedSVMs:
                svm.train(trainFeatures, trainLabels)
            results.extend(svm.validate(svm.predict(testFeatures), testLabels))
        for c in C:
            svm = SVM(C=c, kernel='rbf', pretrained=preTrainedSVMs)
            if not preTrainedSVMs:
                svm.train(trainFeatures, trainLabels)
            results.extend(svm.validate(svm.predict(testFeatures), testLabels))
        results = np.reshape(np.array(results), (20, 8))
        df = pd.DataFrame(results, columns = ['kernel','C','accuracy', 'precision_np', 'precision_p', 'recall_np', 'recall_p', 'f1'])
        joblib.dump(df, './Project/validation_results/svm_testing_results.res')
        print(df)

    
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