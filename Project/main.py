import cv2
import multiprocessing
multiprocessing.set_start_method('forkserver') # For using all cores with sklearn functions
import numpy as np
import warnings 
warnings.filterwarnings("ignore") # Ignore ConvergenceWarnings and the like

from methods.classifiers import KNN, NaiveBayes, NeuralNetwork, SVM
from methods.detector import Detector
from methods.feature_extractor import FeatureExtractor

def main(args = None):

    ############
    # SETTINGS #
    ############
    useNB = False # Use NaiveBayes?
    useKNN = False # Use K-Nearest Neighbors?
    useSVM = False # Use Support Vector Machines?
    useNN = False # Use Neural Networks?
    testDetector = True # Apply the pipeline to a video sequence?
    # To keep consistency between runs
    np.random.seed(0)

    ##################################
    # FEATURE EXTRACTION AND LOADING #
    ##################################
    if useNB or useKNN or useSVM or useNN:
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

    ###################################################
    # BEST CLASSIFIER SHOWCASE ON VIDEO TEST SEQUENCE #
    ###################################################
    if testDetector:   
        detector = Detector(NaiveBayes())
        cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture('./Project/test_sequence/test_sequence.mp4')
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video  file")
        # Read until video is completed
        while(cap.isOpened()):
            
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
            
                detections = detector.process(frame)

                # Display the results after performing NMS
                for (x_tl, y_tl, s, w, h) in detections:
                    # Draw the detections
                    cv2.rectangle(frame, (x_tl, y_tl), (x_tl + w, y_tl + h), (0,0,0))
                # Display the resulting frame
                cv2.imshow('output', frame)     

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break
        # When everything done, release the video capture object
        cap.release()
        
        # Closes all the frames
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()