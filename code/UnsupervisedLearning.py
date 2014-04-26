# Evaluation of Unsupervised Clustering
from sklearn import cluster
import numpy as np
from sklearn import mixture
import Metrics
import Util
import os
import sklearn.metrics

def Eval(X, Y, clstr, return_cluster_labels=False):
	"""
		Inputs:
			X - N by D matrix of data vectors
			Y - N by 1 matrix of true class labels
			clstr - the clustering function 
				either the string = "KMeans" or "GMM"
				or a sklearn clustering instance
					with the methods .fit and 
		Outputs:
			A tuple containing (in the following order):
				Purity Score
				Normalized Mutual Information Score
				Rand Index Score
				(if return_cluster_labels)
					cluster lables for each row in X
	"""

	if type(clstr) == str:
		num_uniq = np.unique(Y).shape[0]
		if clstr.lower() == 'kmeans':
			clstr = cluster.KMeans(n_clusters=num_uniq, max_iter=500, tol=0.0001)
		elif clstr.lower() == 'gmm':
			clstr = mixture.GMM(n_components=num_uniq)

	clstr.fit(X)
	C = clstr.predict(X)

	if return_cluster_labels:
		return (Metrics.Purity(C,Y), sklearn.metrics.normalized_mutual_info_score(Y,C), Metrics.RandIndex(C,Y), C)
	else:
		return (Metrics.Purity(C,Y), sklearn.metrics.normalized_mutual_info_score(Y,C), Metrics.RandIndex(C,Y))


def Run(FeaturesModule, clstr, dirname, train_test='train'):
	(feature_def, X) = FeaturesModule.Features(os.path.join(dirname, train_test))
	if train_test.lower() == 'train':
		Y = Util.LoadClassFile(os.path.join(dirname, 'train_classes.txt'))
	else:
		Y = Util.LoadClassFile(os.path.join(dirname, 'test_classes.txt'))

	return Eval(X,Y, clstr)


