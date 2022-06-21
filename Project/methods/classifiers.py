import joblib
import numpy as np

from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.svm import LinearSVC

class KNN():
	def __init__(self, K, metric, trainFeatures, trainLabels) -> None:
		metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 
		'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 
		'jensenshannon', 'kulsinski', 'kulczynski1', 'mahalanobis', 'matching', 
		'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 
		'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
		if metric not in metrics:
			raise ValueError('Invalid distance metric for K means.')
		if K <= 0:
			raise ValueError('Invalid K value for K means.')
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

	def __init__(self, h, metric, trainFeatures) -> None:
		metrics = ['rect', 'tri', 'gaussian', 'dexp']
		if metric not in metrics:
			raise ValueError('Invalid metric for Parzen Windows.')
		if h <= 0:
			raise ValueError('Invalid h value for Parzen Windows.')
		self.h = h
		self.metric = metric
		self.trainFeatures = trainFeatures

	def gamma(self, x):
		if self.metric == 'rect':
			return 0.5 if abs(x) <= 1 else 0
		elif self.metric == 'tri':
			return 1-abs(x) if abs(x) <= 1 else 0
		elif self.metric == 'gaussian':
			return ((2*np.pi)**(1/2))*np.exp(-0.5*x**2)
		elif self.metric == 'dexp':
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
	def __init__(self, pretrained, modelPath) -> None:
		self.modelPath = modelPath
		self.trained = pretrained
		if self.trained:
			self.load()
		else:
			self.model = LinearSVC()

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