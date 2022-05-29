import os
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def svm(trainData, trainLabels, kernel='linear', max_iter=10000, filename='default'):
    # Initialise the SVM classification model
    model = SVC(kernel=kernel, max_iter=max_iter)
    # Fit the model
    model.fit(trainData, trainLabels)
    # Save the trained model
    if filename == 'default':
        filename = 'trained_' + kernel + 'svm'
    joblib.dump(model, os.path.join('./data', filename + '.sav'))

def getMetrics(model, features, groundTruth):
    predicted = model.predict(features)
    # Calculate confusion matrix using sklearn
    cfs = confusion_matrix(groundTruth,predicted)
    # Manually calculate accuracy, precision and recall (for both classes 6 and 9) from the confusion matrix
    accuracy = np.sum(cfs.diagonal())/np.sum(cfs)
    precision_0 = cfs[0,0]/ np.sum(cfs[:,0])
    precision_1 = cfs[1,1]/ np.sum(cfs[:,1])
    recall_0 = cfs[0,0]/ np.sum(cfs[0,:])
    recall_1 = cfs[1,1]/ np.sum(cfs[1,:])
    return [cfs, accuracy, precision_0, precision_1, recall_0, recall_1]