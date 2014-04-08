import nltk
from nltk.tokenize.punkt import PunktWordTokenizer
import gensim
import logging


def Document2Word2VecTrainingInputFormat(document):
	"""
		Given an input string of plain text sentences, first
		tokenizes the documents into each sentence, then tokenizes
		each sentence at the word level. Returns a list of lists where 
		each inner lists represents a sentence in the input and the contents are the individual words of the sentence.
	"""
	output = list()
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	word_detector = PunktWordTokenizer()
	sentences = sent_detector.tokenize(document)
	for sent in sentences:
		output.append(word_detector.tokenize(sent))
	return output


def ExtractAllSentences(filenames):
	"""
		Input: A list of filenames. 
		Output: A list of lists where each inner list represents a sentence 
		and the items of the list are the words of the sentence
	"""

	output = list()
	for f in filenames:
		try: 
			output = output + (Document2Word2VecTrainingInputFormat(open(f, 'r').read()))
		except:
			None
	return output

def TrainWord2VecOnDocuments(filenames, size=100, alpha=0.025, window=5, min_count=5, seed=1, workers=1, min_alpha=0.0001, sg=1):
	"""
		Trains word2vec on the given files. Returns the model
	"""
	sentences = ExtractAllSentences(filenames)
	model = gensim.models.word2vec.Word2Vec(sentences, size=100, alpha=0.025, window=5, min_count=5, seed=1, workers=1, min_alpha=0.0001, sg=1)
	return model















