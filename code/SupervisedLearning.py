# Evaluation of Supervised Classification
import numpy as np
import Metrics
import Util
import os
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
import sklearn.metrics

def Eval(XTrain, YTrain, XTest, YTest, clf, return_predicted_labels=False):
	"""
		Inputs:
			XTrain - N by D matrix of training data vectors
			YTrain - N by 1 matrix of training class labels
			XTest - M by D matrix of testin data vectors
			YTrain - M by 1 matrix of testing class labels
			clstr - the clustering function 
				either the string = "KMeans" or "GMM"
				or a sklearn clustering instance
					with the methods .fit and 
		Outputs:
			A tuple containing (in the following order):
				Accuracy
				Overall Precision
				Overall Recall
				Overall F1 score
				Avg. Precision per class
				Avg. Recall per class
				F1 Score
				Precision per class
				Recall per class
				F1 Score per class
				(if return_predicted_labels)
					predicted class labels for each row in XTest
	"""

	if type(clf) == str:
		if 'ridge' in clf.lower():
			clf = RidgeClassifier(tol=1e-2, solver="lsqr")
		elif "perceptron" in clf.lower():
			clf = Perceptron(n_iter=50)
		elif "passive aggressive" in clf.lower() or 'passive-aggressive' in clf.lower():
			clf = PassiveAggressiveClassifier(n_iter=50)
		elif 'linsvm' in clf.lower() or 'linearsvm' in clf.lower() or 'linearsvc' in clf.lower():
			clf = LinearSVC()
		elif 'svm' in clf.lower() or 'svc' in clf.lower():
			clf = SVC()
		elif 'sgd' in clf.lower():
			clf = SGDClassifier()
   
	clf.fit(XTrain, YTrain)
	YPred = clf.predict(XTest)


	accuracy = sklearn.metrics.accuracy_score(YTest, YPred)
	(overall_precision, overall_recall, overall_f1, support) = sklearn.metrics.precision_recall_fscore_support(YTest, YPred, average='micro')
	(precision_per_class, recall_per_class, f1_per_class, support_per_class) = sklearn.metrics.precision_recall_fscore_support(YTest, YPred)
	avg_precision_per_class = np.mean(precision_per_class)
	avg_recall_per_class = np.mean(recall_per_class)
	avg_f1_per_class = np.mean(f1_per_class)


	if return_predicted_labels:
		return (accuracy, overall_precision, overall_recall, overall_f1, avg_precision_per_class, avg_recall_per_class, avg_f1_per_class, precision_per_class, recall_per_class, f1_per_class, YPred)
	else:
		return (accuracy, overall_precision, overall_recall, overall_f1, avg_precision_per_class, avg_recall_per_class, avg_f1_per_class, precision_per_class, recall_per_class, f1_per_class)


def Run(FeaturesModule, clf, dirname):
	(feature_def, XTrain) = FeaturesModule.Features(os.path.join(dirname, 'train'))
	XTest = FeaturesModule.Features(os.path.join(dirname, 'test'), feature=feature_def)
	YTrain = Util.LoadClassFile(os.path.join(dirname, 'train_classes.txt'))
	YTest = Util.LoadClassFile(os.path.join(dirname, 'test_classes.txt'))

	return Eval(XTrain, YTrain, XTest, YTest, clf)
