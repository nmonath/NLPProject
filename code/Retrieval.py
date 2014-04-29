# This will be for retrieval
import os
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
import Metrics
import Util
import Word2VecExecuter
import numpy as np 
import EKD
import PAD
from sklearn.metrics.pairwise import euclidean_distances

def Run(FeaturesModule, dirname, TopK=30, train_test='train'):
	(feature_def, X) = FeaturesModule.Features(os.path.join(dirname, train_test))
	if train_test.lower() == 'train':
		Y = Util.LoadClassFile(os.path.join(dirname, 'train_classes.txt'))
	else:
		Y = Util.LoadClassFile(os.path.join(dirname, 'test_classes.txt'))

	kd_tree = KDTree(X, leaf_size=30, metric='euclidean')
	precision = 0
	recall = 0
	for i in range(0, X.shape[0]):
		R = kd_tree.query(kd_tree.data, k=TopK, return_distance=False)[0]
		precision += Metrics.RetrievalPrecision(Y[i], Y[R])
		recall += Metrics.RetrievalRecall(Y[i], Y[R], Y, TopK)
		print("Document " + str(i) + " Matched to" + str(Y[R]) + " Precision: " + str(Metrics.RetrievalPrecision(Y[i], Y[R])) + "Recall: " + str(Metrics.RetrievalRecall(Y[i], Y[R], Y, TopK)))
	precision = precision / X.shape[0]
	recall = recall / X.shape[0]

	return (precision, recall)

def RunExhaustive(FeaturesModule, dirname, TopK=30, train_test='train'):
	(feature_def, X) = FeaturesModule.Features(os.path.join(dirname, train_test))
	if train_test.lower() == 'train':
		Y = Util.LoadClassFile(os.path.join(dirname, 'train_classes.txt'))
	else:
		Y = Util.LoadClassFile(os.path.join(dirname, 'test_classes.txt'))

	N = X.shape[0]
	dists = np.zeros((N,N))
	X = normalize(X)
	for i in xrange(0, N):
		for j in xrange(0, N):
			if not i==j:
				dists[i,j] = euclidean_distances(X[i,:], X[j,:])
	print("**Results**")
	precision = 0
	recall = 0
	for i in xrange(0, N):
		R = dists[i,:].argsort()
		R = R[1:TopK]
		precision += Metrics.RetrievalPrecision(Y[i], Y[R])
		recall += Metrics.RetrievalRecall(Y[i], Y[R], Y, TopK)
		print("Document Number: " + str(i) + " Class: " + str(Y[i]))
		print("Top Matching Documents: " + str(R))
		print("Top Matching Documents (Associated Class Labels): " + str(Y[R]))
		print(" Precision: " + str(Metrics.RetrievalPrecision(Y[i], Y[R])) + " Recall: " + str(Metrics.RetrievalRecall(Y[i], Y[R], Y, TopK)))
	
	precision = precision / N
	recall = recall / N
	print("Average Precision: " + str(precision) + " Average Recall: " + str(recall))
	return (precision, recall)


# def RunEKD(dirname, model, train_test='train',TopK=5):

# 	if train_test.lower() == 'train':
# 		Y = Util.LoadClassFile(os.path.join(dirname, 'train_classes.txt'))
# 	else:
# 		Y = Util.LoadClassFile(os.path.join(dirname, 'test_classes.txt'))







# 	count1 = 0
# 	N = Util.get_num_samples(os.path.join(dirname, train_test))
# 	dists = np.zeros((N,N))
# 	for filename1 in os.listdir(os.path.join(dirname, train_test)):
# 		count2 = 0
# 		if '.srl' in filename1:
# 			Doc1 = EKD.Document(doc_file_name=os.path.join(dirname, train_test, filename1), model=model, use_lemma=False)
# 			for filename2 in os.listdir(os.path.join(dirname, train_test)):
# 				if filename1 == filename2:
# 					dists[count1, count2] = 0
# 					count2 += 1
# 				elif '.srl' in filename2:
# 					Doc2 = EKD.Document(doc_file_name=os.path.join(dirname, train_test, filename2), model=model, use_lemma=False)
# 					dists[count1, count2] = Doc1.distance(Doc2)
# 					count2 += 1
# 			count1 += 1

# 	precision = 0
# 	recall = 0

# 	for i in xrange(0, N):
# 		R = dists[i,:].argsort()
# 		R = R[1:TopK]
# 		precision += Metrics.RetrievalPrecision(Y[i], Y[R])
# 		recall += Metrics.RetrievalRecall(Y[i], Y[R], Y, TopK)
# 		print("Document " + str(i) + " Matched to" + str(Y[R]) + " Precision: " + str(Metrics.RetrievalPrecision(Y[i], Y[R])) + "Recall: " + str(Metrics.RetrievalRecall(Y[i], Y[R], Y, TopK)))
# 	precision = precision / N
# 	recall = recall / N

# 	return (precision, recall)

def RunPAD(dirname, model, train_test, TopK=5):
	if train_test.lower() == 'train':
		Y = Util.LoadClassFile(os.path.join(dirname, 'train_classes.txt'))
	else:
		Y = Util.LoadClassFile(os.path.join(dirname, 'test_classes.txt'))


	count1 = 0
	N = Util.get_num_samples(os.path.join(dirname, train_test))
	dists = np.zeros((N,N))
	for filename1 in os.listdir(os.path.join(dirname, train_test)):
		count2 = 0
		if '.srl' in filename1:
			Doc1 = PAD.Document(doc_file_name=os.path.join(dirname, train_test, filename1), model=model, use_lemma=False)
			for filename2 in os.listdir(os.path.join(dirname, train_test)):
				if filename1 == filename2:
					dists[count1, count2] = 0
					count2 += 1
				elif '.srl' in filename2:
					Doc2 = PAD.Document(doc_file_name=os.path.join(dirname, train_test, filename2), model=model, use_lemma=False)
					dists[count1, count2] = Doc1.distance(Doc2)
					count2 += 1
			count1 += 1

	precision = 0
	recall = 0
	print("\n**Results**")
	for i in xrange(0, N):
		R = dists[i,:].argsort()
		R = R[1:TopK]
		precision += Metrics.RetrievalPrecision(Y[i], Y[R])
		recall += Metrics.RetrievalRecall(Y[i], Y[R], Y, TopK)
		print("Document Number: " + str(i) + " Class: " + str(Y[i]))
		print("Top Matching Documents: " + str(R))
		print("Top Matching Documents (Associated Class Labels: " + str(Y[R]))
		print(" Precision: " + str(Metrics.RetrievalPrecision(Y[i], Y[R])) + " Recall: " + str(Metrics.RetrievalRecall(Y[i], Y[R], Y, TopK)))
	
	precision = precision / N
	recall = recall / N
	print("Average Precision: " + str(precision) + " Average Recall: " + str(recall))
	return (precision, recall)

def RunEKD(dirname, model, train_test='train',TopK=5):
	if train_test.lower() == 'train':
		Y = Util.LoadClassFile(os.path.join(dirname, 'train_classes.txt'))
	else:
		Y = Util.LoadClassFile(os.path.join(dirname, 'test_classes.txt'))




	# # Manditory settings for the Features Module. Others are up to you
	import Features as FeaturesModule
	FeaturesModule.FUNIT = FeaturesModule.FeatureUnits.WORD
	FeaturesModule.FTYPE = FeaturesModule.FeatureType.BINARY
	FeaturesModule.FREP = FeaturesModule.FeatureRepresentation.STRING
	FeaturesModule.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT=False
	FeaturesModule.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME=False
	FeaturesModule.DisplayConfiguration()

	(feature_def, X) = FeaturesModule.Features(os.path.join(dirname, train_test), frep=FeaturesModule.FREP, ftype=FeaturesModule.FTYPE, funit=FeaturesModule.FUNIT)
	(word_idx, emb) = Word2VecExecuter.Word2VecLoadWordsHashTable(model, list(feature_def.T[0]))
	X = FeaturesModule.Features(os.path.join(dirname, train_test), frep=FeaturesModule.FREP, ftype=FeaturesModule.FTYPE, funit=FeaturesModule.FUNIT, feature=np.array(word_idx.keys()).reshape((len(word_idx), 1)))

 	count1 = 0
	N = Util.get_num_samples(os.path.join(dirname, train_test))
	dists = np.zeros((N,N))
	for i in xrange(0, N):
		Doc1 = EKD.Document(embeddings=emb[X[i, :], :]) 
		for j in xrange(0, N):
			if not i == j:
				Doc2 = EKD.Document(embeddings=emb[X[j, :], :])
				dists[i,j] = Doc1.distance(Doc2)

	precision = 0
	recall = 0
	print("\n**Results**")

	for i in xrange(0, N):
		R = dists[i,:].argsort()
		R = R[1:TopK]
		precision += Metrics.RetrievalPrecision(Y[i], Y[R])
		recall += Metrics.RetrievalRecall(Y[i], Y[R], Y, TopK)
		print("Document Number: " + str(i) + " Class: " + str(Y[i]))
		print("Top Matching Documents: " + str(R))
		print("Top Matching Documents (Associated Class Labels: " + str(Y[R]))
		print(" Precision: " + str(Metrics.RetrievalPrecision(Y[i], Y[R])) + " Recall: " + str(Metrics.RetrievalRecall(Y[i], Y[R], Y, TopK)))
	
	precision = precision / N
	recall = recall / N
	print("Average Precision: " + str(precision) + " Average Recall: " + str(recall))
	return (precision, recall)

	
# # This will be for pairwise
# # Compare document to every other document and say if they are the same or different
# def Run(FeaturesModule, dirname, train_test='train', Score Cut off):
# def RunEKD(dirname, train_test='train'):