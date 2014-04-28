from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import os
import Util
import numpy as np
import SupervisedLearning

#######################################################################
#						Specify the classifiers to use                #
#######################################################################

global classifiers
classifiers = ["Ridge", "SVM", "LinearSVM", "Passive Aggressive"]


def run(dirname):



	#######################################################################
	# Load the class labels. These stay the same for the whole experiment #			
	#######################################################################
	global y_train
	global y_test
	y_train = Util.LoadClassFile(os.path.join(dirname, 'train_classes.txt'))
	y_test = Util.LoadClassFile(os.path.join(dirname, 'test_classes.txt'))

	import Features


	##################################################################################################################################
	'''                                            First Run the following Lemmatized Features                                     '''
	'''        BLOCK 1                             Word, Word + DP, Word + PAs, All                                                '''
	'''                                            No Argument Labels    			                                               '''
	'''											   Both Binary & TFIDF 															   '''
	'''											   Distance Function: Euclidean & Normalized Euclidean (Cosine)					   '''
	##################################################################################################################################


	###############################################
	# Settings									  #			
	###############################################
	
	Features.USE_LEMMA = True
	Features.USE_DEP_TAGS = False
	Features.USE_POS_TAGS = False
	Features.USE_ARG_LABELS = False
	Features.USE_MEMORY_MAP = False
	Features.FREP = Features.FeatureRepresentation.HASH
	
	# Words
	Features.FUNIT = Features.FeatureUnits.WORD

	RunClassificationExperiment(Features, dirname)

	# Words & DP
	Features.FUNIT = Features.FeatureUnits.WORDS_AND_DEPENDENCY_PAIRS
	RunClassificationExperiment(Features, dirname)

	# Words & PA
	Features.FUNIT = Features.FeatureUnits.WORDS_AND_PREDICATE_ARGUMENT
	RunClassificationExperiment(Features, dirname)

	# ALL
	Features.FUNIT = Features.FeatureUnits.ALL
	RunClassificationExperiment(Features, dirname)

	# ##################################################################################################################################
	# '''                                            First Run the following Unlemmatized Features                                     '''
	# '''        BLOCK 2                             Word, Word + DP, Word + PAs, All                                                '''
	# '''                                            No Argument Labels    			                                               '''
	# '''											   Both Binary & TFIDF 															   '''
	# '''											   Distance Function: Euclidean & Normalized Euclidean (Cosine)					   '''
	# ##################################################################################################################################


	# ###############################################
	# # Settings									  #			
	# ###############################################
	
	# Features.USE_LEMMA = False
	# Features.CASE_SENSITIVE = False
	# Features.USE_DEP_TAGS = False
	# Features.USE_POS_TAGS = False
	# Features.USE_ARG_LABELS = False
	# Features.USE_MEMORY_MAP = False
	# Features.FREP = Features.FeatureRepresentation.HASH
	
	# # Words
	# Features.FUNIT = Features.FeatureUnits.WORD

	# RunClassificationExperiment(Features, dirname)

	# # Words & DP
	# Features.FUNIT = Features.FeatureUnits.WORDS_AND_DEPENDENCY_PAIRS
	# RunClassificationExperiment(Features, dirname)

	# # Words & PA
	# Features.FUNIT = Features.FeatureUnits.WORDS_AND_PREDICATE_ARGUMENT
	# RunClassificationExperiment(Features, dirname)

	# # ALL
	# Features.FUNIT = Features.FeatureUnits.ALL
	# RunClassificationExperiment(Features, dirname)




def RunClassificationExperiment(FeaturesModule, dirname):

	####################################
	# FV1. Binary - Euclidean Distance #
	# FV2. Binary - Cosine Distance    #
	# FV3. TF-IDF - Euclidean Distance #
	# FV4. TF-IDF - Cosine Distance    #
	####################################


	(feature_def, x_train_count) = FeaturesModule.Features(os.path.join(dirname, 'train'), ftype=FeaturesModule.FeatureType.COUNT)
	x_test_count = FeaturesModule.Features(os.path.join(dirname, 'test'), ftype=FeaturesModule.FeatureType.COUNT, feature=feature_def)

	del feature_def
	# Use sparse matrix to save on memory
	# FV1 & FV2
	x_train_binary = FeaturesModule.ToBINARY(x_train_count)
	x_train_binary_normalized = csr_matrix(normalize(np.float32(x_train_binary)))
	x_train_binary = csr_matrix(x_train_binary)
	x_test_binary = FeaturesModule.ToBINARY(x_test_count)
	x_test_binary_normalized = csr_matrix(normalize(np.float32(x_test_binary)))
	x_test_binary = csr_matrix(x_test_binary)
	# FV3 & FV4 used below
	x_train_tfidf = FeaturesModule.ToTFIDF(x_train_count)
	x_train_tfidf_normalized = csr_matrix(normalize(x_train_tfidf))
	x_train_tfidf = csr_matrix(x_train_tfidf)
	x_test_tfidf = FeaturesModule.ToTFIDF(x_test_count)
	x_test_tfidf_normalized = csr_matrix(normalize(x_test_tfidf))
	x_test_tfidf = csr_matrix(x_test_tfidf)

	del x_test_count
	del x_train_count


	############
	# FV1	   #	
	############
	print('\n')
	print('*' * 80)
	print('*' * 80)



	# The following is a HACK so that everything prints out nicely.
	# If you notice in the call to Features() the type was set to count
	# Then the ToBINARY/ToTFIDF methods are called, this is to save time. The
	# Count matrixes can be used to create both binary and tf-idf and so
	# this makes it so we only need to extract the features once for each representation
	FeaturesModule.FTYPE = FeaturesModule.FeatureType.BINARY
	# Print configuration fo FV1
	FeaturesModule.DisplayConfiguration()
	ScoreClassifiers(x_train_binary, y_train, x_test_binary, y_test, "Euclidean")
	
	print('*' * 80)
	print('*' * 80)
	print('\n')


	del x_train_binary
	del x_test_binary


	############
	# FV2	   #	
	############
	print('\n')
	print('*' * 80)
	print('*' * 80)



	# The following is a HACK so that everything prints out nicely.
	# If you notice in the call to Features() the type was set to count
	# Then the ToBINARY/ToTFIDF methods are called, this is to save time. The
	# Count matrixes can be used to create both binary and tf-idf and so
	# this makes it so we only need to extract the features once for each representation
	FeaturesModule.FTYPE = FeaturesModule.FeatureType.BINARY
	# Print configuration fo FV2
	FeaturesModule.DisplayConfiguration()
	ScoreClassifiers(x_train_binary_normalized, y_train, x_test_binary_normalized, y_test, "Cosine")
	
	print('*' * 80)
	print('*' * 80)
	print('\n')

	del x_train_binary_normalized
	del x_test_binary_normalized

	############
	# FV3	   #	
	############
	print('\n')
	print('*' * 80)
	print('*' * 80)



	# The following is a HACK so that everything prints out nicely.
	# If you notice in the call to Features() the type was set to count
	# Then the ToBINARY/ToTFIDF methods are called, this is to save time. The
	# Count matrixes can be used to create both binary and tf-idf and so
	# this makes it so we only need to extract the features once for each representation
	FeaturesModule.FTYPE = FeaturesModule.FeatureType.TFIDF
	# Print configuration fo FV3
	FeaturesModule.DisplayConfiguration()
	ScoreClassifiers(x_train_tfidf, y_train, x_test_tfidf, y_test, "Euclidean")
	
	print('*' * 80)
	print('*' * 80)
	print('\n')

	del x_train_tfidf
	del x_test_tfidf

	############
	# FV4	   #	
	############

	print('\n')
	print('*' * 80)
	print('*' * 80)



	# The following is a HACK so that everything prints out nicely.
	# If you notice in the call to Features() the type was set to count
	# Then the ToBINARY/ToTFIDF methods are called, this is to save time. The
	# Count matrixes can be used to create both binary and tf-idf and so
	# this makes it so we only need to extract the features once for each representation
	FeaturesModule.FTYPE = FeaturesModule.FeatureType.TFIDF
	# Print configuration fo FV4
	FeaturesModule.DisplayConfiguration()
	ScoreClassifiers(x_train_tfidf_normalized, y_train, x_test_tfidf_normalized, y_test, "Cosine")
	
	print('*' * 80)
	print('*' * 80)
	print('\n')

	del x_train_tfidf_normalized
	del x_test_tfidf_normalized

	








def ScoreClassifiers(XTrain, YTrain, XTest, YTest, DistanceFunctionName):

	print('%' * 40 + "\n")


	for clf in classifiers:
		print("** Classifier: " + clf)
		print("** Distance Function: " + DistanceFunctionName)
		(accuracy, overall_precision, overall_recall, overall_f1, avg_precision_per_class, avg_recall_per_class, avg_f1_per_class, precision_per_class, recall_per_class, f1_per_class) = SupervisedLearning.Eval(XTrain, YTrain, XTest, YTest, clf)

		print("\nThe following are the results of classification:")
		print("\nAccuracy: " + str(accuracy))
		print("\nOverall Precision: " + str(overall_precision))
		print("\nOverall Recall: " + str(overall_recall))
		print("\nOverall F1: " + str(overall_f1))
		print("\nAverage Precision Per Class: " + str(avg_precision_per_class))
		print("\nAverage Recall Per Class: " + str(avg_recall_per_class))
		print("\nAverage F1 Per Class: " + str(avg_f1_per_class))
		print("\nPrecision Per Class: " + str(precision_per_class))
		print("\nRecall Per Class: " + str(recall_per_class))
		print("\nF1 Per Class: " + str(f1_per_class))

		print('%' * 40 + "\n")






run("../data_sets/brown_corpus/")


















