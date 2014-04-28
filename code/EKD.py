from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
import Word2VecExecuter
import numpy as np
import Features

class Document:

	def __init__(self, embeddings=None, doc_file_name=None, word_index=None, model=None, use_lemma=False):
		# Normal case, build kdtree right from embeddings:
		if (embeddings== None and (not word_index == None) and (not model == None)):
			(idx, embeddings) = Word2VecExecuter.Word2VecLoadWordsHashTable(model, word_index)
			embeddings = np.array(embeddings)
		elif ((not doc_file_name == None) and (not model == None)):
			Features.USE_LEMMA = use_lemma
			Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
			Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
			words = Features.ReadDependencyParseFile(doc_file_name, funit=Features.FeatureUnits.WORD, remove=False)
			(word_index, embeddings) = Word2VecExecuter.Word2VecLoadWordsHashTable(model, words)
			embeddings = np.array(embeddings)
			del word_index

		self.kd_tree = KDTree(normalize(embeddings), leaf_size=30, metric='euclidean')


	def distance(self, other, theta=0.5):
		if other.__class__ == Document:
			(d_self_to_other, i_self_to_other) = self.kd_tree.query(other.kd_tree.data, k=1, return_distance=True) 
			del i_self_to_other
			(d_other_to_self, i_other_to_self) = other.kd_tree.query(self.kd_tree.data, k=1, return_distance=True) 
			del i_other_to_self
			return np.mean(d_self_to_other)*theta + np.mean(d_other_to_self)*(1-theta)