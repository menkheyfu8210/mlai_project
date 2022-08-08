import numpy as np
import warnings 
warnings.filterwarnings("ignore")

from methods.classifiers import KNN, NaiveBayes, NeuralNetwork, SVM
from methods.feature_extractor import FeatureExtractor
from sklearn.neighbors import KNeighborsClassifier
import joblib

def main(args = None):

    ############
    # SETTINGS #
    ############
    useNB = True
    useKNN = True
    useSVM = True
    useNN = True
    doHardNegativeMining = True
    # To keep consistency between runs
    np.random.seed(0)

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
        nb = NaiveBayes()
        if not nb.trained:
            # Determine the best parameter set for the classifier
            nb.hyperparameter_tuning(trainFeatures, trainLabels)
            # Train the best model
            nb.train(trainFeatures, trainLabels)
        # Validate the model against the test features
        nb.validate(nb.predict(testFeatures), testLabels)

    #########################
    # CLASSIFICATION W/ KNN #
    #########################
    if useKNN:
        knn = KNN()
        if not knn.trained:
            # Determine the best parameter set for the classifier
            knn.hyperparameter_tuning(trainFeatures, trainLabels)
            # Train the best model
            knn.train(trainFeatures, trainLabels)
        # Validate the model against the test features
        knn.validate(knn.predict(testFeatures), testLabels)

    #########################
    # CLASSIFICATION W/ SVM #
    #########################
    if useSVM:
        svm = SVM()
        if not svm.trained:
            # Determine the best parameter set for the classifier
            svm.hyperparameter_tuning(trainFeatures, trainLabels)
            # Train the best model
            svm.train(trainFeatures, trainLabels)
        # Validate the model against the test features
        svm.validate(svm.predict(testFeatures), testLabels)

    ####################################
    # CLASSIFICATION W/ NEURAL NETWORK #
    ####################################
    if useNN:
        nn = NeuralNetwork()
        if not nn.trained:
            # Determine the best parameter set for the classifier
            nn.hyperparameter_tuning(trainFeatures, trainLabels)
            # Train the best model
            nn.train(trainFeatures, trainLabels)
        # Validate the model against the test features
        nn.validate(nn.predict(testFeatures), testLabels)

    ######################################
    # RETRAINING W/ HARD-NEGATIVE MINING #
    ######################################
    if doHardNegativeMining:
        print('Starting retraining with hard-negative mining.')
        # Extract the additional features if they haven't been already
        if not fe.preExtractedAddFeat:
            fe.extract_add_features
        # Load the additional features
        addFeatures = fe.load_add_features()
        # NaiveBayes
        nb = NaiveBayes()
        if nb.trained:
            if not nb.retrained:
                nb.retrain(trainFeatures, trainLabels, addFeatures)
            # Validate the model against the test features
            nb.validate(nb.predict(testFeatures), testLabels)
        else:
            print(nb.model_name + ' was not trained, skipping.')
        # KNN
        knn = KNN()
        if nb.trained:
            if not knn.retrained:
                knn.retrain(trainFeatures, trainLabels, addFeatures)
                # Validate the model against the test features
            knn.validate(knn.predict(testFeatures), testLabels)
        else:
            print(knn.model_name + ' was not trained, skipping.')
        # SVM
        svm = SVM()
        if svm.trained:
            if not svm.retrained:
                svm.retrain(trainFeatures, trainLabels, addFeatures)
            # Validate the model against the test features
            svm.validate(svm.predict(testFeatures), testLabels)
        else:
            print(svm.model_name + ' was not trained, skipping.')
        # NeuralNetwork
        nn = NeuralNetwork()
        if nn.trained:
            if not nn.retrained:
                nn.retrain(trainFeatures, trainLabels, addFeatures)
            # Validate the model against the test features
            nn.validate(nn.predict(testFeatures), testLabels)
        else:
            print(nn.model_name + ' was not trained, skipping.')


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