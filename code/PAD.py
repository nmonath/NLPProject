from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from sklearn.metrics import euclidean_distances
import Word2VecExecuter
import numpy as np
from Features import Word
import Features
import copy

class Document:

	def __init__(self, pred_arg_structures=None, embeddings=None, doc_file_name=None, word_index=None, model=None, use_lemma=True):
		self.pred_arg_structures = pred_arg_structures
		self.use_lemma = use_lemma
		if (embeddings== None and doc_file_name==None and (not word_index == None) and (not model == None)):
			(word_index, embeddings) = Word2VecExecuter.Word2VecLoadWordsHashTable(model, word_index)
			embeddings = np.array(embeddings)
		elif (not doc_file_name == None) and (not model == None):
			Features.USE_LEMMA = use_lemma
			words = Features.ReadDependencyParseFile(doc_file_name, funit=Features.FeatureUnits.WORD, remove=False)
			(word_index, embeddings) = Word2VecExecuter.Word2VecLoadWordsHashTable(model, words)
			embeddings = np.array(embeddings)
			self.pred_arg_structures = Features.ReadDependencyParseFile(doc_file_name, funit=Features.FeatureUnits.PRED_ARG, remove=False)
			del words
		self.embeddings = normalize(np.array(embeddings))
		self.word_index = word_index

	def word_string(self, word):
		if self.use_lemma:
			return word.lemma
		else:
			return word.form

	def distance(self, other, theta=0.5):
		dist_self_to_other = 0
		for pa in self.pred_arg_structures:
			dist_self_to_other = dist_self_to_other + self.padist(pa, other)
		dist_self_to_other = dist_self_to_other/len(self.pred_arg_structures)

		dist_other_to_self = 0
		for pa in other.pred_arg_structures:
			dist_other_to_self = dist_other_to_self + other.padist(pa, self)
		dist_other_to_self = dist_other_to_self/len(other.pred_arg_structures)

		return dist_self_to_other*theta + dist_other_to_self*(1-theta)

	def padist(self, pa_in_self, other, return_pa=False, penalty=2):
		mindist = 0;
		minpa = ""
		for pa_other in other.pred_arg_structures:
			dist = 0;
			terms_compared = 0;
			try:
				dist = dist + euclidean_distances(self.embeddings[self.word_index[self.word_string(pa_in_self.pred)]], other.embeddings[other.word_index[other.word_string(pa_other.pred)]])
				terms_compared = terms_compared + 1
			except:
				dist = dist + penalty
				terms_compared = terms_compared + 1

			for arg_label_self in pa_in_self.args:
				emb_arg_self = None
				try:
					for w in pa_in_self.args[arg_label_self]:
						if emb_arg_self:
							try:
								emb_arg_self = emb_arg_self + self.embeddings[self.word_index[self.word_string(w)]]
							except:
								None
						else:
							try:
								emb_arg_self = self.embeddings[self.word_index[self.word_string(w)]]
							except:
								None
				except:
					None


				if not (emb_arg_self == None):
					terms_compared = terms_compared + 1

					if arg_label_self in pa_other.args:
						emb_arg_other = None
						try:
							for w in pa_other.args[arg_label_self]:
								if emb_arg_other:
									try:
										emb_arg_other = emb_arg_other + other.embeddings[other.word_index[other.word_string(w)]]
									except:
										None
								else:
									try:
										emb_arg_other = other.embeddings[other.word_index[other.word_string(w)]]
									except:
										None
							dist = dist + euclidean_distances(emb_arg_self, emb_arg_other)
						except:
							dist = dist + penalty
							terms_compared = terms_compared + 1

				else:
					dist = dist + penalty
					terms_compared = terms_compared + 1
			if terms_compared > 0:
				dist = dist/terms_compared
			else:
				dist = 2

			if dist < mindist:
				mindist = dist
				minpa = copy.deepcopy(pa_other)
		if return_pa:
			return (mindist, minpa)
		else:
			return mindist


