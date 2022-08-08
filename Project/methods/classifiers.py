import joblib
import numpy as np
import os

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm

class Classifier():
	"""Parent class for all classifier models. All children differ only in the
	init method.

    Parameters
    ----------
	parameter_space : dict, default={}
		Dictionary containing the parameter space for hyperparameter tuning.

    model_path : str, default='./models/'
        Path where the model is stored.

    model_name : str, default='unnamed'
        Specifies the model's name for debug purposes.

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
				estimator=None,
				parameter_space={},
				model_path='./Project/models/', 
				model_name='unnamed') -> None:
		self.estimator = estimator
		self.parameter_space = parameter_space
		self.model_name = model_name
		self.model_path = model_path
		if not os.path.exists(model_path):
			os.makedirs(model_path)
		self.trained = os.path.exists(self.model_path + self.model_name)
		self.retrained = os.path.exists(self.model_path + 'retrained_' + self.model_name)
		self.loaded = False

	def load(self, model_path, model_name):
		"""Load a pretrained model.
        """
		if self.trained:
			self.model = joblib.load(model_path + model_name)
			self.loaded = True
			return self.model
		else: 
			raise FileNotFoundError(model_path + model_name + ': file not found.')  

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
		if not self.trained:
			# Train the model on the provided data
			trained_model = self.model.fit(train_features, train_labels)
			# Save the model for future use
			joblib.dump(trained_model, self.model_path + self.model_name)
		else:
			self.load(self.model_path, self.model_name)
		self.trained = True

	def retrain(self, train_features, train_labels, add_features):
		"""Rerain the model after performing hard-negative mining.

        Parameters
        ----------
        train_features : array-like of shape (n_samples, n_features)
            Training features, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        train_labels : array-like of shape (n_samples,)
            Class labels associated to the training features.

        add_features : array-like of shape (n_samples, n_features)
            Additional features, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        """
		if not self.loaded:
			self.load()
		# Perform hard-negative mining
		pbar = tqdm(add_features)
		for feature in pbar:
			pbar.set_description('Performing hard negative mining with ' + self.model_name)
			# If the model misclassifies, add that feature vector to the training vector
			if self.model.predict(feature) == 1:
				train_features.append(feature)
				train_labels.append(0)
		# Retrain
		retrained_model = self.model.fit(train_features, train_labels)
		# Save the model for future use
		joblib.dump(retrained_model, self.model_path + 'retrained_' + self.model_name)
		
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
		if self.trained:
			return self.model.predict(test_features)
		else: 
			raise RuntimeError('Model not trained.')  

	def validate(self, prediction, test_labels):
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
		dict containing a summary of the precision, recall, F1 score for each class
        """
		print('Validation results:')
		print(classification_report(test_labels, prediction, target_names=['Non-pedestrian', 'Pedestrian']))
	
	def hyperparameter_tuning(self, train_features, train_labels):
		"""Perform a grid search with cross validation on the parameter space in
		order to determine the optimal hyperparameter set.
		
        Parameters
        ----------
        train_features : array-like of shape (n_samples, n_features)
            Training features, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        train_labels : array-like of shape (n_samples,)
            Class labels associated to the training features.
        """
		print('Starting hyperparameter tuning for ' +  self.model_name)
		clf = GridSearchCV(self.model, self.parameter_space, cv=3, verbose=10)
		clf.fit(train_features, train_labels)
		# Best parameter set
		print('Best parameters found:\n', clf.best_params_)
		# All results
		print('Results for all runs:')
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
		self.model = clf.best_estimator_

class NaiveBayes(Classifier):
	"""Naive Bayes based classification. This is a wrapper class around
	sklearn's GaussianNB class.

    Parameters
    ----------
	retrained : bool, default=False
		Specifies whether or not to load the retrained model

    model_path : str, default='./models/naive_bayes/'
        Specifies where to save and look for trained models.

    Attributes
    ----------
	model : object
		Instance of sklearn.naive_bayes.GaussianNB
    """

	def __init__(self, retrained=False, model_path='./Project/models/naive_bayes/') -> None:
		print(f"Initializing NaiveBayes")
		self.model_path = model_path
		if retrained:
			self.model_name = 'retrained_NaiveBayes.mod'
		else:
			self.model_name = 'NaiveBayes.mod'
		self.trained = os.path.exists(model_path + self.model_name)
		if self.trained: 
			self.model = super().load(self.model_path, self.model_name)
		else:
			# Initialize the model
			self.model = GaussianNB()
		# Set up the space for hyperparameter tuning
		parameter_space = {
			'var_smoothing': np.logspace(0,-9, num=50)
		}
		super().__init__(self.model, parameter_space, model_path, self.model_name)

class KNN(Classifier):
	"""K-Nearest Neighbors based classification. This is a wrapper class around
	sklearn's KNeighborsClassifier class.

    Parameters
    ----------
	retrained : bool, default=False
		Specifies whether or not to load the retrained model

    model_path : str, default='./models/knn/'
        Specifies where to save and look for trained models.

    Attributes
    ----------		
	model : object
		Instance of sklearn.neighbors.KNeighborsClassifier
    """

	def __init__(self, retrained=False, model_path='./Project/models/knn/') -> None:
		print(f"Initializing KNN")
		self.model_path = model_path
		if retrained:
			self.model_name = 'retrained_KNearestNeighbors.mod'
		else:
			self.model_name = 'KNearestNeighbors.mod'
		self.trained = os.path.exists(model_path + self.model_name)
		if self.trained: 
			self.model = super().load(self.model_path, self.model_name)
		else:
			# Initialize the model
			self.model = KNeighborsClassifier()	
		# Set up the space for hyperparameter tuning
		parameter_space = {
			'n_neighbors' : [1, 3, 5, 7, 9, 11, 25, 51, 75, 101],
			'p' : [1, 2]
		}
		super().__init__(self.model, parameter_space, model_path, self.model_name)	

class SVM(Classifier):
	"""Support Vector Machine based classification. This is a wrapper class around
	sklearn's SVC class.

    Parameters
    ----------
	retrained : bool, default=False
		Specifies whether or not to load the retrained model

    model_path : str, default='./models/svm/'
        Specifies where to save and look for trained models.

    Attributes
    ----------		
	model : object
		Instance of sklearn.svm.SVC
    """

	def __init__(self, retrained=False, model_path='./Project/models/svm/', ) -> None:
		print(f"Initializing SVM")
		self.model_path = model_path
		if retrained:
			self.model_name = 'retrained_SupportVectorMachine.mod'
		else:
			self.model_name = 'SupportVectorMachine.mod'
		self.trained = os.path.exists(model_path + self.model_name)
		if self.trained: 
			self.model = super().load(self.model_path, self.model_name)
		else:
			# Initialize the model
			self.model = SVC()
		# Set up the space for hyperparameter tuning
		parameter_space = {
			'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
			'kernel' : ['linear', 'poly', 'sigmoid', 'rbf']
		}
		super().__init__(self.model, parameter_space, model_path, self.model_name)	

class NeuralNetwork(Classifier):
	"""Neural network based classification. This is a wrapper class around
	sklearn's MLPClassifier class.

    Parameters
    ----------
	retrained : bool, default=False
		Specifies whether or not to load the retrained model

    model_path : str, default='./models/neural_network/'
        Specifies where to save and look for trained models.

    Attributes
    ----------
	model : object
		Instance of sklearn.neural_network.MLPClassifier
    """

	def __init__(self, retrained=False, model_path='./Project/models/neural_network/') -> None:
		print(f"Initializing NeuralNetwork")
		self.model_path = model_path
		if retrained:
			self.model_name = 'retrained_NeuralNetwork.mod'
		else:
			self.model_name = 'NeuralNetwork.mod'
		self.trained = os.path.exists(model_path + self.model_name)
		if self.trained: 
			self.model = super().load(self.model_path, self.model_name)
		else:
			# Initialize the model
			self.model = MLPClassifier()
		# Set up the space for hyperparameter tuning
		parameter_space = {
			'alpha' : [0.0001, 0.001, 0.01, 0.1],
			'hidden_layer_sizes' : [(750,), (500,), (250,), (750, 325), (500, 250), (250, 125),]

		}
		super().__init__(self.model, parameter_space, model_path, self.model_name)	