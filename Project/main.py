import joblib
import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")

from methods.classifiers import SVM, KNN, NaiveBayes, NeuralNetwork
from methods.feature_extractor import FeatureExtractor
from tqdm import tqdm

def main(args = None):

    ############
    # SETTINGS #
    ############
    useNB = False
    preTrainedNB = True
    useKNN = False
    preTrainedKNNs = True
    useSVMs = False
    preTrainedSVMs = True
    useNN = True
    preTrainedNN = False

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
        print('Naive Bayes testing results:')
        print(df)

    #########################
    # CLASSIFICATION W/ KNN #
    #########################
    if useKNN:
        # Initialize a bunch of KNN classifiers with varying K values and distance metrics
        pbar = tqdm([1, 5, 11, 21, 51, 75, 101, 251, 501, 1001, 2501, 5001, 7501, 10001, 15001, 19001])
        results = []
        for k in pbar:
            knn = KNN(k, 'cityblock', pretrained=preTrainedKNNs)
            if not preTrainedKNNs:
                pbar.set_description('Training Cityblock KNN w/ K=' + str(k))
                knn.train(trainFeatures, trainLabels)
            pbar.set_description('Testing Cityblock KNN w/ K=' + str(k))
            results.extend(knn.validate(knn.predict(testFeatures), testLabels))
        pbar = tqdm([1, 5, 11, 21, 51, 75, 101, 251, 501, 1001, 2501, 5001, 7501, 10001, 15001, 19001])
        for k in pbar:
            knn = KNN(k, 'euclidean', pretrained=preTrainedKNNs)
            if not preTrainedKNNs:
                pbar.set_description('Training Euclidean KNN w/ K=' + str(k))
                knn.train(trainFeatures, trainLabels)
            pbar.set_description('Testing Euclidean KNN w/ K=' + str(k))
            results.extend(knn.validate(knn.predict(testFeatures), testLabels))
        results = np.reshape(np.array(results), (32, 8))
        df = pd.DataFrame(results, columns = ['metric','K','accuracy', 'precision_np', 'precision_p', 'recall_np', 'recall_p', 'f1'])
        joblib.dump(df, './Project/validation_results/knn_testing_results.res')
        print('KNN testing results:')
        print(df)
        
    #########################
    # CLASSIFICATION W/ SVM #
    #########################
    # Train (or load) a bunch of SVMs with different kernels and varying 
    # regularization params, then validate each one
    if useSVMs:
        pbar = tqdm([0.01, 0.1, 1, 10, 100])
        results = []
        for c in pbar:
            svm = SVM(C=c, kernel='linear', pretrained=preTrainedSVMs)
            if not preTrainedSVMs:
                pbar.set_description('Training Linear SVM w/ C=' + str(c))
                svm.train(trainFeatures, trainLabels)
            pbar.set_description('Testing Linear SVM w/ C=' + str(c))
            results.extend(svm.validate(svm.predict(testFeatures), testLabels))
        pbar = tqdm([0.01, 0.1, 1, 10, 100])
        for c in pbar:
            svm = SVM(C=c, kernel='poly', pretrained=preTrainedSVMs)
            if not preTrainedSVMs:
                pbar.set_description('Training Poly SVM w/ C=' + str(c))
                svm.train(trainFeatures, trainLabels)
            pbar.set_description('Testing Poly SVM w/ C=' + str(c))
            results.extend(svm.validate(svm.predict(testFeatures), testLabels))
        pbar = tqdm([0.01, 0.1, 1, 10, 100])
        for c in pbar:
            svm = SVM(C=c, kernel='sigmoid', pretrained=preTrainedSVMs)
            if not preTrainedSVMs:
                pbar.set_description('Training Sigmoid SVM w/ C=' + str(c))
                svm.train(trainFeatures, trainLabels)
            pbar.set_description('Testing Sigmoid SVM w/ C=' + str(c))
            results.extend(svm.validate(svm.predict(testFeatures), testLabels))
        pbar = tqdm([0.01, 0.1, 1, 10, 100])
        for c in pbar:
            svm = SVM(C=c, kernel='rbf', pretrained=preTrainedSVMs)
            if not preTrainedSVMs:
                pbar.set_description('Training RBF SVM w/ C=' + str(c))
                svm.train(trainFeatures, trainLabels)
            pbar.set_description('Testing RBF SVM w/ C=' + str(c))
            results.extend(svm.validate(svm.predict(testFeatures), testLabels))
        results = np.reshape(np.array(results), (20, 8))
        df = pd.DataFrame(results, columns = ['kernel','C','accuracy', 'precision_np', 'precision_p', 'recall_np', 'recall_p', 'f1'])
        joblib.dump(df, './Project/validation_results/svm_testing_results.res')
        print(df)

    ####################################
    # CLASSIFICATION W/ NEURAL NETWORK #
    ####################################
    if useNN:
        results = []
        nb = NeuralNetwork(pretrained=preTrainedNN)
        if not preTrainedNN:
            nb.train(trainFeatures, trainLabels)
        results.extend(nb.validate(nb.predict(testFeatures), testLabels))
        results = np.reshape(np.array(results), (1, 8))
        df = pd.DataFrame(results, columns = ['-','-','accuracy', 'precision_np', 'precision_p', 'recall_np', 'recall_p', 'f1'])
        joblib.dump(df, './Project/validation_results/nn_testing_results.res')
        print('Neural Network testing results:')
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