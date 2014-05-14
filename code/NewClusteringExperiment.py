import json
from sklearn.svm import LinearSVC
import Features
import Util
import sys
from scipy.sparse import csr_matrix
import numpy as np
import copy
import os
from sklearn.preprocessing import normalize
import sklearn.metrics
from sklearn import cluster
import Metrics




def RunClusteringExperiment(FeaturesJSON, ClustererJSON):

	# Load the classification settings
	classifier_settings =  json.load(open(ClustererJSON))
	# Error handling
	if not 'Dataset' in classifier_settings:
		raise(BaseException('No Dataset was defined in the classifier settings'))


	# Set the configurations for the Features Module
	Features.SetConfigurations(FeaturesJSON)

	# Load the Class Label files
	y_train = Util.LoadClassFile(os.path.join(classifier_settings['Dataset'], 'train_classes.txt'))
	y_labels = Util.LoadClassLabels(os.path.join(classifier_settings['Dataset'], 'class_label_index.txt'))
	num_uniq = np.unique(y_train).shape[0]

	clusterers = [cluster.KMeans(n_clusters=num_uniq, max_iter=500, tol=0.0001)]

	# Extract the features from training documents
	(feature_def, x_train_count) = Features.Features(os.path.join(classifier_settings['Dataset'], 'train'), ftype=Features.FeatureType.COUNT)

	# Binary Features
	x_train_binary = Features.ToBINARY(x_train_count)
	x_train_binary_normalized = csr_matrix(normalize(np.float64(x_train_binary)), dtype=np.float64)
	del x_train_binary

	ScoreClustering('Binary', x_train_binary_normalized, y_train, y_labels, copy.deepcopy(clusterers))
	del x_train_binary_normalized

	# tf-idf Features
	x_train_tfidf = Features.ToTFIDF(x_train_count)
	x_train_tfidf_normalized = csr_matrix(normalize(x_train_tfidf), dtype=np.float64)
	del x_train_tfidf

	ScoreClustering('tf-idf', x_train_tfidf_normalized, y_train, y_labels, copy.deepcopy(clusterers))
	del x_train_tfidf_normalized


	# Count Features
	x_train_count_normalized = csr_matrix(normalize(np.float64(x_train_count)), dtype=np.float64)
	del x_train_count
	ScoreClustering('Count', x_train_count_normalized, y_train, y_labels, copy.deepcopy(clusterers))
	del x_train_count_normalized










def ScoreClustering(FeatureType, XTrain, YTrain, YLabels, clusterers):

	# Record 1
	# FeatureUnit, FeatureType, Lemmatized, Case Sensitive, POSTags, DepTags, PredArgTags, Clustering Algorithm, Clustering Param, Purity, NMI, ARI

	
	record_1 = ""
	num_uniq = np.unique(YTrain).shape[0]


	print("\n\n")
	for cst in clusterers:

		cst.fit(XTrain)
		YPred = cst.predict((XTrain))

		# Clear the clustere
		cst = 0

		# Obtain the results
		purity = Metrics.Purity(YTrain,YPred)
		NMI = sklearn.metrics.normalized_mutual_info_score(YTrain,YPred)
		ARI = sklearn.metrics.adjusted_rand_score(YTrain, YPred)
		
		# Print the results
		record_1 += (Features.FUNIT + " & " + FeatureType + " & " + str(Features.USE_LEMMA) + " & " 
			+ str(Features.CASE_SENSITIVE) + " & " + str(Features.USE_POS_TAGS) + " & " + str(Features.USE_DEP_TAGS) 
			+ " & " + str(Features.USE_ARG_LABELS) + " & " + "k-means" + " & " + "k="+ str(num_uniq) 
			+ " & " + str(purity) + " & " + str(NMI) + " & " + str(ARI) + " \\\\ \n")


	print(record_1)
	print("\n" + '%' * 40 + "\n")


RunClusteringExperiment(sys.argv[1], sys.argv[2])

