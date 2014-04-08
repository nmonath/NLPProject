import nltk
from nltk.tokenize.punkt import PunktWordTokenizer
import gensim
import logging
from os import listdir
from os.path import isfile, isdir, join
from copy import copy


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

def GetAllFileNamesFromDir(d): 
	if isdir(d):
			onlyfiles = [ join(d,fi) for fi in listdir(d) if isfile(join(d,fi)) and not fi.startswith('.') ]
	return onlyfiles
		

def TrainWord2VecOnDocuments(filenames_or_directory, size=100, alpha=0.025, window=5, min_count=5, seed=1, workers=1, min_alpha=0.0001, sg=1):
	"""
		Trains word2vec on the given files. Returns the model
		Input can be a list of filenames or a list of directories
		or both. It will be trained on all filenames and all the files
		in the directories
	"""

	filenames = list()
	for f in filenames_or_directory:
		if isdir(f):
			filenames = filenames + copy(GetAllFileNamesFromDir(f))
		else:
			filenames.append(f)
	sentences = ExtractAllSentences(filenames)
	model = gensim.models.word2vec.Word2Vec(sentences, size=size, alpha=alpha, window=window, min_count=min_count, seed=seed, workers=workers, min_alpha=min_alpha, sg=sg)
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	return model















