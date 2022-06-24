import joblib
import numpy as np
import time

from math import nan
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC

class Classifier():
	"""Parent class for all classifier models.

    Parameters
    ----------
    model_name : str, default='unnamed'
        Specifies the model's name for debug purposes.

    string : str, default='unnamed'
        Model-specific string (e.g. kernel type for SVM, distance metric for \
		KNN, etc.)

    parameter : float, default=nan
        Model-specific parameter (e.g. C for SVM, K for KNN, etc.)

    Attributes
    ----------
    model_name : str
        Specifies the model's name.

    string : str
        Model-specific string (e.g. kernel type for SVM, distance metric for KNN
		, etc.)

    parameter : float
        Model-specific parameter (e.g. C for SVM, K for KNN, etc.)
    """

	def __init__(self, model_name='unnamed', string='', parameter=nan) -> None:
		self.model_name = model_name
		self.string = string
		self.parameter = parameter

	def _validate(self, prediction, test_labels, disp=False):
		"""Validate the prediction of a model against the known testing data's
		class labels.
		
        Parameters
        ----------
        prediction : array-like of shape (n_samples,)
            Predicted class labels for the testing data.

	    test_labels : array-like of shape (n_samples,)
            Actual class labels for the testing data.

		Returns
		-------
		numpy array of shape (1, 7) holding:

		- str, Model-specific (e.g. kernel type for SVM, distance metric for KNN
		- float, Model-specific (e.g. C for SVM, K for KNN, etc.)
		- float, accuracy
		- float, precision w.r.t. class 0
		- float, precision w.r.t. class 1
		- float, recall w.r.t. class 0
		- float, recall w.r.t. class 1
        """
		accuracy = round(accuracy_score(test_labels, prediction) * 100, 3)
		precision_np = round(precision_score(test_labels, prediction, pos_label=0), 3)
		precision_p = round(precision_score(test_labels, prediction, pos_label=1), 3)
		recall_np = round(recall_score(test_labels, prediction, pos_label=0), 3)
		recall_p = round(recall_score(test_labels, prediction, pos_label=1), 3)
		cfs = confusion_matrix(test_labels, prediction)
		ret = np.array([self.string, self.parameter, accuracy, precision_np, precision_p, recall_np, recall_p]).T
		if disp:
			print("Model: " + self.model_name)
			print(f"Classifier accuracy: {(accuracy)}%")
			print(f"Precision w.r.t class non-pedestrian: {precision_np}")
			print(f"Precision w.r.t class pedestrian: {precision_p}")
			print(f"Recall w.r.t class non-pedestrian: {recall_np}")
			print(f"Recall w.r.t class pedestrian: {recall_p}")
			print('Confusion matrix:\n')
			print(cfs)
		return ret
	
class SVM(Classifier):
	"""Support Vector Machine based classification.

    The implementation is based on sklearn's SVC. 

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
        Specifies the kernel type to be used in the algorithm.

    model_path : str, default='./models/'
        Specifies where to save and look for trained models.

	pretrained : bool, default=True
		Specifies whether or not the SVM was already trained.

	debug : bool, default=True
		Specifies whether or not to print debug information to the console

    Attributes
    ----------
    C : float
        Regularization parameter. 

    kernel : str
        Kernel type used in the algorithm.

    max_iter : int
        Hard limit on iterations within solver.
		
    model_name : str
        Name of the model, obtained as: kernel + C + 'SVM.mod'

    model_path : str
        Full path of where the trained model is saved.

	trained : bool
		True if the SVM was already trained, False otherwise
		
	debug : bool
		True if debug information is printed to the console, False otherwise 
    """

	def __init__(self, 
			 	C=1.0, 
				kernel='rbf', 
				max_iter=1000,
				model_path='./models/', 
				pretrained=False, 
				debug=True) -> None:
		# Input argument checks
		kernels = ['linear', 'poly', 'rbf', 'sigmoid']
		if C <= 0:
			raise ValueError('C value must be strictly positive.')
		if max_iter <= 0: 
			raise ValueError('Maximum iteration number must be strictly positive.')
		if kernel not in kernels:
			raise ValueError('Kernel type not recognized. Possible options are: ' +
				'"linear", "poly", "rbf", "sigmoid".')
		self.debug = debug
		# Keep the SVM parameters saved
		self.C = C
		self.kernel = kernel
		self.max_iter = max_iter
		# Path to save/load the model to/from
		self.model_name = kernel + 'C' + str(C).replace('.','') + 'SVM.mod'
		self.model_path = model_path + self.model_name
		self.trained = pretrained
		if self.trained:
			# Load the SVM
			if self.debug: print("Loading " + self.model_path)
			self.load()
		else:
			# Initialize the SVM
			if self.debug: print("Initializing " + kernel + f" SVM w/ C={self.C}")
			self.model = SVC(C=self.C, kernel=self.kernel, max_iter=self.max_iter)
		super().__init__(self.model_name, self.kernel, self.C)

	def train(self, train_features, train_labels):
		"""Train the SVM model on the given training data.

        Parameters
        ----------
        train_features : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        train_labels : array-like of shape (n_samples,)
            Class labels associated to the training vectors.
        """
		if not self.trained:
			if self.debug: 
				print("Training " + self.model_name)
				start_time = time.time()
			# Train the SVM on the provided data
			self.model.fit(train_features, train_labels)
			if self.debug: 
				elapsed = time.time() - start_time
				print(f"Finished training (time elapsed: {elapsed}s), saving to: {self.model_path}")
			# Save the model for future use
			joblib.dump(self.model, self.modelPath)
			self.trained = True
		else:
			print('Model was already trained, skipping training.')

	def predict(self, test_features):
		"""Make a prediction on testing data.

        Parameters
        ----------
        test_features : array-like of shape (n_samples, n_features)
            Testing vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

		Returns
		-------
		array-like of shape (n_samples,) holding the prediction for each testing
		vector.
        """
		if self.trained:
			return self.model.predict(test_features)
		else:
			raise RuntimeError('SVM was not yet trained.')

	def load(self):
		"""Load a pretrained model.
        """
		if self.trained:
			self.model = joblib.load(self.model_path)
		else: 
			raise RuntimeError('SVM was not yet trained.')       

	def validate(self, test_features, test_labels, disp=False):
		# Make a prediction on the test features
		prediction = self.predict(test_features)
		super()._validate(self, prediction, test_labels, disp)

# 
class KNN():
	def __init__(self, K, metric) -> None:
		metrics = ['cityblock', 'euclidean', 'minkowski']
		if metric not in metrics:
			raise ValueError('Metric not recognized. Possible options are: ' +
				'"cityblock, "euclidean", "minkowski"')
		if K <= 0:
			raise ValueError('K value must be strictly positive.')
		self.K = K
		self.metric = metric

	def train(self, trainFeatures, testFeatures):
		if self.metric == 'minkowski':
			self.D = cdist(trainFeatures, testFeatures, metric=self.metric, p=3.)
		else:
			self.D = cdist(trainFeatures, testFeatures, metric=self.metric)

	def predict(self, trainLabels):
		neighbors = np.argsort(self.D, axis=0)
		kNeighbors = neighbors[:self.K, :]
		neighborsLabels = np.array(trainLabels)[kNeighbors]
		return stats.mode(neighborsLabels, axis=0)[0].flatten()

	def validate(self, trainLabels, testLabels, _print=False):
		prediction = self.predict(trainLabels)
		accuracy = round(accuracy_score(testLabels, prediction) * 100, 3)
		precision_np = round(precision_score(testLabels, prediction, pos_label=0), 3)
		precision_p = round(precision_score(testLabels, prediction, pos_label=1), 3)
		recall_np = round(recall_score(testLabels, prediction, pos_label=0), 3)
		recall_p = round(recall_score(testLabels, prediction, pos_label=1), 3)
		cfs = confusion_matrix(testLabels, prediction)
		ret = np.array([self.metric, self.K, accuracy, precision_np, precision_p, recall_np, recall_p]).T
		if _print:
			print('Model: ' + self.modelPath)
			print(f"Classifier accuracy: {(accuracy)}%")
			print(f"Precision w.r.t class non-pedestrian: {precision_np}")
			print(f"Precision w.r.t class pedestrian: {precision_p}")
			print(f"Recall w.r.t class non-pedestrian: {recall_np}")
			print(f"Recall w.r.t class pedestrian: {recall_p}")
			print('Confusion matrix:\n')
			print(cfs)
		return ret
class ParzenWindows():

	def __init__(self, h, kernel, trainFeatures, trainLabels) -> None:
		kernels = ['rect', 'tri', 'gaussian', 'dexp']
		if kernel not in kernels:
			raise ValueError('Kernel type not recognized. Possible options are: ' +
				'"rect", "tri", "gaussian", "dexp".')
		if h <= 0:
			raise ValueError('h value must be strictly positive.')
		self.h = h
		self.kernel = kernel
		self.trainClass0 = trainFeatures[trainLabels == 0]
		self.trainClass1 = trainFeatures[trainLabels == 1]

	def gamma(self, x):
		if self.kernel == 'rect':
			return 0.5 if abs(x) <= 1 else 0
		elif self.kernel == 'tri':
			return 1-abs(x) if abs(x) <= 1 else 0
		elif self.kernel == 'gaussian':
			return ((2*np.pi)**(1/2))*np.exp(-0.5*x**2)
		elif self.kernel == 'dexp':
			return 0.5*np.exp(-abs(x))
		else:
			raise ValueError('Kernel type not recognized. Possible options are: ' +
				'"rect", "tri", "gaussian", "dexp".')

	def predict(self, testFeatures):
		predicted = []
		for x_te in testFeatures[:,0]:
			g1, g2 = [], []
			for x0 in self.trainClass0[:,0]:
				g1 = np.append(g1, self.gamma((x_te - x0)/self.h, self.kernel))
			for x1 in self.trainClass1[:,0]:
				g2 = np.append(g2, self.gamma((x_te - x1)/self.h, self.kernel))
			l0 = np.mean(g1)/self.h # Likelihood of class 0
			l1 = np.mean(g2)/self.h # Likelihood of class 1
			predicted = np.append(predicted, 0 if l0 > l1 else 1)
		return predicted

	def validate(self, testFeatures, testLabels, _print=False):
		prediction = self.predict(testFeatures)
		accuracy = round(accuracy_score(testLabels, prediction) * 100, 3)
		precision_np = round(precision_score(testLabels, prediction, pos_label=0), 3)
		precision_p = round(precision_score(testLabels, prediction, pos_label=1), 3)
		recall_np = round(recall_score(testLabels, prediction, pos_label=0), 3)
		recall_p = round(recall_score(testLabels, prediction, pos_label=1), 3)
		cfs = confusion_matrix(testLabels, prediction)
		ret = np.array([self.kernel, self.C, accuracy, precision_np, precision_p, recall_np, recall_p]).T
		if _print:
			print('Model: ' + self.modelPath)
			print(f"Classifier accuracy: {(accuracy)}%")
			print(f"Precision w.r.t class non-pedestrian: {precision_np}")
			print(f"Precision w.r.t class pedestrian: {precision_p}")
			print(f"Recall w.r.t class non-pedestrian: {recall_np}")
			print(f"Recall w.r.t class pedestrian: {recall_p}")
			print('Confusion matrix:\n')
			print(cfs)
		return ret