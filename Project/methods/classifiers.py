import joblib
import numpy as np
import os
import time

from math import nan
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.svm import SVC

class Classifier():
	"""Parent class for all classifier models.

    Parameters
    ----------
    model_path : str, default='./models/'
        Path where the model is stored.

    model_name : str, default='unnamed'
        Specifies the model's name for debug purposes.

    string : str, default='unnamed'
        Model-specific string (e.g. kernel type for SVM, distance metric for \
		KNN, etc.)

    parameter : float, default=nan
        Model-specific parameter (e.g. C for SVM, K for KNN, etc.)

    Attributes
    ----------
    model_path : str
        Model save location.

    model_name : str
        Specifies the model's name.

    string : str
        Model-specific string (e.g. kernel type for SVM, distance metric for \
		KNN, etc.).

    parameter : float
        Model-specific parameter (e.g. C for SVM, K for KNN, etc.).

	loaded : bool
		True if the model has been loaded, False otherwise.
    """

	def __init__(self, 
				model_path='./models/', 
				model_name='unnamed', 
				string='', 
				parameter=nan) -> None:
		self.model_name = model_name
		self.model_path = model_path
		if not os.path.exists(model_path):
			os.makedirs(model_path)
		self.string = string
		self.parameter = parameter
		self.loaded = False

	def load(self):
		"""Load a pretrained model.
        """
		if os.path.exists(self.model_path + self.model_name):
			self.model = joblib.load(self.model_path + self.model_name)
			self.loaded = True
		else: 
			raise FileNotFoundError(self.model_path + self.model_name + ': file not found.')  

	def train(self, train_features, train_labels):
		"""Train the model on the given training data.

        Parameters
        ----------
        train_features : array-like of shape (n_samples, n_features)
            Training features, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        train_labels : array-like of shape (n_samples,)
            Class labels associated to the training features.
        """
		if not os.path.exists(self.model_path + self.model_name):
			print("Training " + self.model_name)
			start_time = time.time()
			# Train the model on the provided data
			trained_model = self.model.fit(train_features, train_labels)
			elapsed = time.time() - start_time
			print(f"Finished training (time elapsed: {elapsed}s), saving to: {self.model_path + self.model_name}")
			# Save the model for future use
			joblib.dump(trained_model, self.model_path + self.model_name)
		else:
			print('Model was already trained, skipping training.')

	def predict(self, test_features):
		"""Make a prediction on testing data.

        Parameters
        ----------
        test_features : array-like of shape (n_samples, n_features)
            Testing features, where `n_samples` is the number of samples
            and `n_features` is the number of features.

		Returns
		-------
		array-like of shape (n_samples,) holding the prediction for each testing
		feature.
        """
		if os.path.exists(self.model_path + self.model_name):
			start_time = time.time()
			prediction = self.model.predict(test_features)
			elapsed = time.time() - start_time
			print(f"Finished predicting (time elapsed: {elapsed}s).")
			return prediction
		else: 
			raise RuntimeError('Model not trained.')  

	def validate(self, prediction, test_labels, disp=False):
		"""Validate the prediction of the model against the known testing data's
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

		- str, Model-specific (e.g. kernel type for SVM, distance metric for KNN)
		- float, Model-specific (e.g. C for SVM, K for KNN, etc.)
		- float, accuracy
		- float, precision w.r.t. class 0
		- float, precision w.r.t. class 1
		- float, recall w.r.t. class 0
		- float, recall w.r.t. class 1
        """
		if not self.loaded:
			self.load()
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
	"""Support Vector Machine based classification. This is a wrapper class around
	sklearn's SVC class.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
        Specifies the kernel type to be used in the algorithm.

    max_iter : int, default=1000
        Hard limit on iterations within solver. -1 for no limit.

    model_path : str, default='./models/svm/'
        Specifies where to save and look for trained models.

	pretrained : bool, default=True
		Specifies whether or not the SVM was already trained.

    Attributes
    ----------		
	model : object
		Instance of sklearn.svm.SVC
    """

	def __init__(self, 
			 	C=1.0, 
				kernel='rbf', 
				max_iter=10000,
				model_path='./models/svm/', 
				pretrained=False) -> None:
		# Input arguments checks
		kernels = ['linear', 'poly', 'rbf', 'sigmoid']
		if C <= 0: 
			raise ValueError('C value must be strictly positive.')
		if max_iter <= 0:  
			max_iter = -1
		if kernel not in kernels:
			raise ValueError('Kernel type not recognized. Possible options are: ' +
				'"linear", "poly", "rbf", "sigmoid".')
		model_name = kernel + 'C' + str(C).replace('.','') + 'SVM.mod'
		super().__init__(model_path, model_name, kernel, C)
		if pretrained: 
			super().load()
		else:
			# Initialize the model
			print("Initializing " + kernel + f" SVM w/ C={C}")
			self.model = SVC(C=C, kernel=kernel, max_iter=max_iter)

class KNN(Classifier):
	"""K-Nearest Neighbors based classification. This is a wrapper class around
	sklearn's KNeighborsClassifier class.

    Parameters
    ----------
    K : int,
        Number of neighbors to be considered in classification.

    metric : {'cityblock', 'euclidean', 'minkowski'}, default='euclidean'
        Specifies the distance metric to be used in the algorithm.

    model_path : str, default='./models/knn/'
        Specifies where to save and look for trained models.

	pretrained : bool, default=True
		Specifies whether or not the model was already trained.

    Attributes
    ----------		
	model : object
		Instance of sklearn.neighbors.KNeighborsClassifier
    """

	def __init__(self, 
				K, 
				metric='euclidean', 
				model_path='./models/knn/', 
				pretrained=False) -> None:
		# Input arguments checks
		metrics = ['cityblock', 'euclidean', 'minkowski']
		if metric not in metrics:
			raise ValueError('Metric not recognized. Possible options are: ' +
				'"cityblock, "euclidean", "minkowski"')
		if K <= 0:
			raise ValueError('K value must be strictly positive.')
		p_dict = {
			'cityblock' : 1, 
			'euclidean' : 2, 
			'minkowski' : 3
		}
		model_name = metric + 'K' + str(K) + 'KNN.mod'
		super().__init__(model_path, model_name, metric, K)
		if pretrained: 
			super().load()
		else:
			# Initialize the model
			print("Initializing " + metric + f" KNN w/ K={K}")
			self.model = KNeighborsClassifier(n_neighbors=K, 
											p=p_dict[metric], 
											metric=metric)

class NaiveBayes(Classifier):
	"""Naive Bayes based classification. This is a wrapper class around
	sklearn's GaussianNB class.

    Parameters
    ----------
    model_path : str, default='./models/naive_bayes/'
        Specifies where to save and look for trained models.

	pretrained : bool, default=True
		Specifies whether or not the model was already trained.

    Attributes
    ----------
	model : object
		Instance of sklearn.naive_bayes.GaussianNB
    """

	def __init__(self,  
				model_path='./models/naive_bayes/', 
				pretrained=False) -> None:
		model_name = 'NaiveBayes.mod'
		if pretrained: 
			super().load(model_path, model_name, nan, '')
		else:
			# Initialize the model
			print(f"Initializing NaiveBayes")
			self.model = GaussianNB()
		super().__init__(model_path, model_name, '', nan)