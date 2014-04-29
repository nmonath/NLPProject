from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import os
import Util
import numpy as np
import UnsupervisedLearning

#######################################################################
#			  Specify the Clustering algorithms to use                #
#######################################################################

global clusterers
clusterers = ["KMeans"]


def run(dirname, train_test='train'):



	#######################################################################
	# Load the class labels. These stay the same for the whole experiment #			
	#######################################################################
	global y
	if train_test.lower() == 'train':
		y = Util.LoadClassFile(os.path.join(dirname, 'train_classes.txt'))
	else:
		y = Util.LoadClassFile(os.path.join(dirname, 'test_classes.txt'))

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

	RunClusteringExperiment(Features, dirname, train_test)

	# Words & DP
	Features.FUNIT = Features.FeatureUnits.WORDS_AND_DEPENDENCY_PAIRS
	RunClusteringExperiment(Features, dirname, train_test)

	# Words & PA
	Features.FUNIT = Features.FeatureUnits.WORDS_AND_PREDICATE_ARGUMENT
	RunClusteringExperiment(Features, dirname, train_test)

	# ALL
	Features.FUNIT = Features.FeatureUnits.ALL
	RunClusteringExperiment(Features, dirname, train_test)

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

	# RunClusteringExperiment(Features, dirname)

	# # Words & DP
	# Features.FUNIT = Features.FeatureUnits.WORDS_AND_DEPENDENCY_PAIRS
	# RunClusteringExperiment(Features, dirname)

	# # Words & PA
	# Features.FUNIT = Features.FeatureUnits.WORDS_AND_PREDICATE_ARGUMENT
	# RunClusteringExperiment(Features, dirname)

	# # ALL
	# Features.FUNIT = Features.FeatureUnits.ALL
	# RunClusteringExperiment(Features, dirname)




def RunClusteringExperiment(FeaturesModule, dirname, train_test):

	####################################
	# FV1. Binary - Euclidean Distance #
	# FV2. Binary - Cosine Distance    #
	# FV3. TF-IDF - Euclidean Distance #
	# FV4. TF-IDF - Cosine Distance    #
	####################################


	(feature_def, x_count) = FeaturesModule.Features(os.path.join(dirname, train_test), ftype=FeaturesModule.FeatureType.COUNT)

	del feature_def
	# Use sparse matrix to save on memory
	# FV1 & FV2
	x_binary = FeaturesModule.ToBINARY(x_count)
	x_binary_normalized = (normalize(np.float32(x_binary)))
	x_binary = (x_binary)
	
	# FV3 & FV4 used below
	x_tfidf = FeaturesModule.ToTFIDF(x_count)
	x_tfidf_normalized = (normalize(x_tfidf))
	x_tfidf = (x_tfidf)
	
	del x_count


	############
	# FV1	   #	
	############
	# print('\n')
	# print('*' * 80)
	# print('*' * 80)



	# # The following is a HACK so that everything prints out nicely.
	# # If you notice in the call to Features() the type was set to count
	# # Then the ToBINARY/ToTFIDF methods are called, this is to save time. The
	# # Count matrixes can be used to create both binary and tf-idf and so
	# # this makes it so we only need to extract the features once for each representation
	# FeaturesModule.FTYPE = FeaturesModule.FeatureType.BINARY
	# # Print configuration fo FV1
	# FeaturesModule.DisplayConfiguration()
	# ScoreClusterer(x_binary, y, "Euclidean")
	
	# print('*' * 80)
	# print('*' * 80)
	# print('\n')


	del x_binary


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
	ScoreClusterer(x_binary_normalized, y, "Cosine")
	
	print('*' * 80)
	print('*' * 80)
	print('\n')

	del x_binary_normalized

	############
	# FV3	   #	
	# ############
	# print('\n')
	# print('*' * 80)
	# print('*' * 80)



	# # The following is a HACK so that everything prints out nicely.
	# # If you notice in the call to Features() the type was set to count
	# # Then the ToBINARY/ToTFIDF methods are called, this is to save time. The
	# # Count matrixes can be used to create both binary and tf-idf and so
	# # this makes it so we only need to extract the features once for each representation
	# FeaturesModule.FTYPE = FeaturesModule.FeatureType.TFIDF
	# # Print configuration fo FV3
	# FeaturesModule.DisplayConfiguration()
	# ScoreClusterer(x_tfidf, y, "Euclidean")
	
	# print('*' * 80)
	# print('*' * 80)
	# print('\n')

	del x_tfidf

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
	ScoreClusterer(x_tfidf_normalized, y, "Cosine")
	
	print('*' * 80)
	print('*' * 80)
	print('\n')

	del x_tfidf_normalized

	








def ScoreClusterer(X, Y, DistanceFunctionName):

	print('%' * 40 + "\n")


	for clstr in clusterers:
		print("** Clustering Algorithm: " + clstr)
		print("** Distance Function: " + DistanceFunctionName)

		(purity, mutual_info_score, rand_index) = UnsupervisedLearning.Eval(X,Y,clstr)

		print("\nThe following are the results of unsupervised classification:")
		print("\nPurity: " + str(purity))
		print("\nNormalized Mutual Information Score: " + str(mutual_info_score))
		print("\nRand Index: " + str(rand_index))

		print('%' * 40 + "\n")






run("../data_sets/brown_clustering/", train_test="train")


















