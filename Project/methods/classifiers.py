import joblib
import numpy as np

from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC

class KNN():
	def __init__(self, K, metric, trainFeatures, trainLabels) -> None:
		metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 
		'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 
		'jensenshannon', 'kulsinski', 'kulczynski1', 'mahalanobis', 'matching', 
		'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 
		'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
		if metric not in metrics:
			raise ValueError('Metric not recognized. Possible options are: ' +
				'"braycurtis", "canberra", "chebyshev", "cityblock', 
		'correlation", "cosine", "dice", "euclidean", "hamming", "jaccard', 
		'jensenshannon", "kulsinski", "kulczynski1", "mahalanobis", "matching', 
		'minkowski", "rogerstanimoto", "russellrao", "seuclidean', 
		'sokalmichener", "sokalsneath", "sqeuclidean", "yule"')
		if K <= 0:
			raise ValueError('K value must be strictly positive.')
		self.K = K
		self.metric = metric
		self.trainFeatures = trainFeatures
		self.trainLabels = trainLabels

	def predict(self, feature):
		D = cdist(self.trainFeatures, feature, metric=self.metric)
		neighbors = np.argsort(D, axis=0)
		kNeighbors = neighbors[:self.K, :]
		neighborsLabels = np.array(self.trainLabels)[kNeighbors]
		return [stats.mode(neighborsLabels, axis=0)[0].flatten(), 1]

class ParzenWindows():

	def __init__(self, h, kernel, trainFeatures) -> None:
		kernels = ['rect', 'tri', 'gaussian', 'dexp']
		if kernel not in kernels:
			raise ValueError('Kernel type not recognized. Possible options are: ' +
				'"rect", "tri", "gaussian", "dexp".')
		if h <= 0:
			raise ValueError('h value must be strictly positive.')
		self.h = h
		self.kernel = kernel
		self.trainFeatures = trainFeatures

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

	def predict(self, feature):
		g1, g2 = [], []
		for x1 in self.trainFeatures:
			g1 = np.append(g1, self.gamma((feature - x1)/self.h))
		l1 = np.mean(g1)/self.h # Likelihood of class 0
		l2 = np.mean(g2)/self.h # Likelihood of class 1
		return [0 if l1 > l2 else 1, 1]

class SVM():
	def __init__(self, C=1.0, kernel='linear', maxIter=1000, modelPath='./models/', pretrained=False) -> None:
		kernels = ['linear', 'poly', 'rbf', 'sigmoid']
		if C <= 0:
			raise ValueError('C value must be strictly positive.')
		if maxIter <= 0:
			maxIter = -1
		if kernel not in kernels:
			raise ValueError('Kernel type not recognized. Possible options are: ' +
				'"linear", "poly", "rbf", "sigmoid".')
		
		self.modelPath = modelPath + kernel + 'C' + str(C).replace('.','') + 'SVM.mod'
		self.trained = pretrained
		if self.trained:
			self.load()
		else:
			self.model = SVC(C=C, kernel=kernel, max_iter=maxIter)

	def train(self, features, labels):
		if not self.trained:
			self.model.fit(features, labels)
			joblib.dump(self.model, self.modelPath)
			self.trained = True
		else:
			print('Model was already trained, skipping training...')

	def predict(self, feature):
		return [self.model.predict(feature), self.model.decision_function(feature)]

	def load(self):
		if self.trained:
			self.model = joblib.load(self.modelPath)
		else: 
			raise Exception('SVM was not yet trained.')       

	def validate(self, testFeatures, testLabels, _print=False):
		prediction = self.model.predict(testFeatures)
		accuracy = accuracy_score(testLabels, prediction)
		precision_np = precision_score(testLabels, prediction, pos_label=0)
		precision_p = precision_score(testLabels, prediction, pos_label=1)
		recall_np = recall_score(testLabels, prediction, pos_label=0)
		recall_p = recall_score(testLabels, prediction, pos_label=1)
		cfs = confusion_matrix(testLabels, prediction)
		if _print:
			print('Model: ' + self.modelPath)
			print('Classifier accuracy: ' + "{0:.2f}".format(accuracy * 100) + '%')
			print('Precision w.r.t class non-pedestrian: ' + "{0:.2f}".format(precision_np))
			print('Precision w.r.t class pedestrian: ' + "{0:.2f}".format(precision_p))
			print('Recall w.r.t class non-pedestrian: ' + "{0:.2f}".format(recall_np))
			print('Recall w.r.t class pedestrian: ' + "{0:.2f}".format(recall_p))
			print('Confusion matrix:\n')
			print(cfs)