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




def RunClassificationExperiment(FeaturesJSON, ClassifierJSON):

	# Load the classification settings
	classifier_settings =  json.load(open(ClassifierJSON))
	# Error handling
	if not 'C' in classifier_settings:
		classifier_settings['C'] = [0.1, 0.5, 1.0]
	if not 'Dataset' in classifier_settings:
		raise(BaseException('No Dataset was defined in the classifier settings'))

	# Instatiate the classifiers
	classifiers = [LinearSVC(C=c) for c in classifier_settings['C']]

	# Set the configurations for the Features Module
	Features.SetConfigurations(FeaturesJSON)

	

	# Load the Class Label files
	y_train = Util.LoadClassFile(os.path.join(classifier_settings['Dataset'], 'train_classes.txt'))
	y_test = Util.LoadClassFile(os.path.join(classifier_settings['Dataset'], 'test_classes.txt'))
	y_labels = Util.LoadClassLabels(os.path.join(classifier_settings['Dataset'], 'class_label_index.txt'))


	# Extract the features from training documents
	(feature_def, x_train_count) = Features.Features(os.path.join(classifier_settings['Dataset'], 'train'), ftype=Features.FeatureType.COUNT)

	# Binary Features
	x_train_binary = Features.ToBINARY(x_train_count)
	x_train_binary_normalized = csr_matrix(normalize(np.float32(x_train_binary)))
	del x_train_binary

	# tf-idf Features
	x_train_tfidf = Features.ToTFIDF(x_train_count)
	x_train_tfidf_normalized = csr_matrix(normalize(x_train_tfidf))
	del x_train_tfidf

	# Count Features
	x_train_count_normalized = csr_matrix(normalize(np.float32(x_train_count)))
	del x_train_count


	# Extract Features from testing documents
	x_test_count = Features.Features(os.path.join(classifier_settings['Dataset'], 'test'), ftype=Features.FeatureType.COUNT, feature=feature_def)
	del feature_def

	# Binary
	x_test_binary = Features.ToBINARY(x_test_count)
	x_test_binary_normalized = csr_matrix(normalize(np.float32(x_test_binary)))
	del x_test_binary

	ScoreClassifiers('Binary', x_train_binary_normalized, y_train, x_test_binary_normalized,  y_test, y_labels, copy.deepcopy(classifiers))
	del x_test_binary_normalized
	del x_train_binary_normalized


	x_test_tfidf = Features.ToTFIDF(x_test_count)
	x_test_tfidf_normalized = csr_matrix(normalize(x_test_tfidf))
	del x_test_tfidf 


	ScoreClassifiers('tf-idf', x_train_tfidf_normalized, y_train, x_test_tfidf_normalized,  y_test, y_labels, copy.deepcopy(classifiers))
	del x_test_tfidf_normalized
	del x_train_tfidf_normalized

	x_test_count_normalized = csr_matrix(normalize(np.float32(x_test_count)))
	del x_test_count

	ScoreClassifiers('Count', x_train_count_normalized, y_train, x_test_count_normalized,  y_test, y_labels, copy.deepcopy(classifiers))
	del x_test_count_normalized
	del x_train_count_normalized




def ScoreClassifiers(FeatureType, XTrain, YTrain, XTest, YTest, YLabels, classifiers):

	# Record 1
	# FeatureUnit, FeatureType, Lemmatized, Case Sensitive, POSTags, DepTags, PredArgTags, Classifier, Classifier Param, Accuracy, Micro F1, Macro F1

	# Record 2
	# FeatureUnit, FeatureType, Lemmatized, Case Sensitive, POSTags, DepTags, PredArgTags, Classifier, Classifier Param, Class Name & Num Samples & Precision & Recall & F1 


	record_1 = ""
	record_2 = ""


	print("\n\n")
	for clf in classifiers:

		# Train the classifier and classify test cases
		clf.fit(XTrain, YTrain)
		YPred = clf.predict(XTest)

		# Record the value of C
		C = clf.C

		# Clear the classifier
		clf = 0

		# Obtain the results
		accuracy = sklearn.metrics.accuracy_score(YTest, YPred)
		f1_micro = sklearn.metrics.f1_score(YTest, YPred, average='micro') 
		f1_macro = sklearn.metrics.f1_score(YTest, YPred, average='macro') 
		(precision_per_class, recall_per_class, f1_per_class, support_per_class) = sklearn.metrics.precision_recall_fscore_support(YTest, YPred)

		# Print the results
		record_1 += (Features.FUNIT + " & " + FeatureType + " & " + str(Features.USE_LEMMA) + " & " 
			+ str(Features.CASE_SENSITIVE) + " & " + str(Features.USE_POS_TAGS) + " & " + str(Features.USE_DEP_TAGS) 
			+ " & " + str(Features.USE_ARG_LABELS) + " & " + "Linear SVM" + " & " + "C="+ str(C) 
			+ " & " + str(accuracy) + " & " + str(f1_micro) + " & " + str(f1_macro) + " \\\\ \n")

		for i in range(0, len(YLabels)):
			record_2 += (Features.FUNIT + " & " + FeatureType + " & " + str(Features.USE_LEMMA) + " & " 
				+ str(Features.CASE_SENSITIVE) + " & " + str(Features.USE_POS_TAGS) + " & " + str(Features.USE_DEP_TAGS) 
				+ " & " + str(Features.USE_ARG_LABELS) + " & " + "Linear SVM" + " & " + "C="+ str(C) 
				+ " & " + YLabels[i] + " & " + str(support_per_class[i]) + " & " + str(precision_per_class[i]) + " & " + str(recall_per_class[i]) + " & " + str(f1_per_class[i]) + " \\\\ \n")


	print(record_1)
	print(record_2)		
	print("\n\n")


RunClassificationExperiment(sys.argv[1], sys.argv[2])

