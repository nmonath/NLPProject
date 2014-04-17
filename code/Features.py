import numpy as np
import os
import tempfile
import sys
from scipy.sparse import csr_matrix
from Util import *
from enum import Enum

global USE_LEMMA 
USE_LEMMA = True

def DataType(argin):
		if argin == FeatureType.BINARY:
			return np.bool
		elif argin == FeatureType.TFIDF:
			return np.float16
		elif argin == FeatureType.COUNT:
			return np.uint32
		elif argin == FeatureRepresentation.HASH:
			return np.dtype(int)
		elif argin == FeatureRepresentation.STRING:
			return np.object

def ConversionFunction(argin):
		if argin == FeatureRepresentation.HASH:
			return hash
		elif argin == FeatureRepresentation.STRING:
			return str
	
class FeatureUnits(Enum):
	WORD = 'WORD'
	DEPENDENCY_PAIR = 'DP'
	BOTH = 'BOTH'

class FeatureType(Enum):
	BINARY = 'BINARY'
	TFIDF = 'tf-idf'
	COUNT = 'COUNT'

class FeatureRepresentation(Enum):
	HASH = 'HASH'
	STRING = 'STRING'

class Word:
	"""
		A Word object consists of the following fields:
			form - the way the word appears in the document
			lemma - the lemmatized form of the word
			posTag - the part of speech tag for the word
			feats - the features specified in the dep file, _ == no features
			depRel - the dependency label of the word

		Two words are considered equal if the lemmatized forms are equal
	"""

	def __init__(self, form, lemma, posTag, feats, depRel):
		self.form = form
		self.lemma = lemma
		self.posTag = posTag
		self.feats = feats
		self.depRel = depRel

	def __eq__(self, other):
		return self.lemma == other.lemma

	def __ne__(self, other):
		return not self.eq(other)

	def __str__(self):
		return self.lemma if USE_LEMMA else self.form

	def __hash__(self):
		return hash(self.lemma) if USE_LEMMA else hash(self.form)

class Dependency:
	"""
		A Dependency object consists of a list of words 
		and the sentence that the dependency appears in
		the sentence numbers start at 0.

		Two dependencies are equal if their heads and complements are equal
	"""
	def __init__(self, head, complement, sentenceNo):
		self.head = head
		self.complement = complement
		self.sentenceNo = sentenceNo

	def __eq__(self, other):
		return self.head == other.head and self.complement == other.complement

	def __ne__(self, other):
		return not self.eq(other)

	def fancy_string(self):
		s = "{" + str(self.head)+ ", " + str(self.complement) +  "} -- Sentence #" + str(self.sentenceNo) 
		return s

	def __str__(self):
		return self.head.lemma + " " + self.complement.lemma if USE_LEMMA else self.head.form + " " + self.complement.form

	def __hash__(self):
		return (hash(self.head.lemma + " " + self.complement.lemma)) if USE_LEMMA else (hash(self.head.form + " " + self.complement.form)) 

def ReadDependencyParseFile(filename, funit=FeatureUnits.BOTH):
	""" 
		This function reads a dep format file into a (python) list of Word and or Dependency objects.
	"""
	Words = list()
	Dependencies = list()

	if funit == FeatureUnits.WORD or funit == FeatureUnits.BOTH:
		f = open(filename, 'r')
		# Mapping from (sentence number, head id number) to a list of the complements of the head id number
		sentenceNo = 0
		for line in f:
			spl = line.split()
			if len(spl) == 7 or len(spl) == 8:
				wordno = int(spl[0])-1
				wordform = spl[1]
				lemma = spl[2]
				posTag = spl[3]
				feat = spl[4]
				head = int(spl[5])-1
				depRel = spl[6]
				Words.append(Word(wordform, lemma, posTag, feat, depRel))
		if funit == FeatureUnits.WORD: 
			return Words
	if funit == FeatureUnits.DEPENDENCY_PAIR or funit == FeatureUnits.BOTH:
		f = open(filename, 'r')
		# Mapping from (sentence number, head id number) to a list of the complements of the head id number
		ComplementsOfHeadInSentence = dict()
		WordsInSentence = dict()
		WordsInSentence[0] = []
		sentenceNo = 0
		for line in f:
			spl = line.split()
			if len(spl) == 7 or len(spl) == 8:
				wordno = int(spl[0])-1
				wordform = spl[1]
				lemma = spl[2]
				posTag = spl[3]
				feat = spl[4]
				head = int(spl[5])-1
				depRel = spl[6]
				WordsInSentence[sentenceNo].append(Word(wordform, lemma, posTag, feat, depRel))
				# 
				if not head == -1:
					if (sentenceNo, head) in ComplementsOfHeadInSentence:
						ComplementsOfHeadInSentence[(sentenceNo, head)].append(wordno)
					else:
						ComplementsOfHeadInSentence[(sentenceNo, head)] = [wordno]
			else:
				sentenceNo = sentenceNo + 1
				WordsInSentence[sentenceNo] = []
		for (sentno, headno) in ComplementsOfHeadInSentence:
			for compno in ComplementsOfHeadInSentence[(sentno, headno)]:
				#print str(sentno) + " " + str(headno) + " " + str(compno)
				#Display(WordsInSentence[sentno])
				Dependencies.append(Dependency(WordsInSentence[sentno][headno], WordsInSentence[sentno][compno], sentno))
		
		if funit == FeatureUnits.DEPENDENCY_PAIR:
			return Dependencies
		elif funit == FeatureUnits.BOTH:
			return Dependencies + Words

def Features(dirname, funit=FeatureUnits.WORD, ftype=FeatureType.BINARY, frep=FeatureRepresentation.HASH, feature=None, K=0.5, UseLemma=True):
	"""
		Creates an M-by-N matrix where N is the length of the feature vector and M is number of documents
		The documents used are all the .srl files stored in the directory dirname
	"""		

	USE_LEMMA = UseLemma

	f_data_type = DataType(ftype);

	num_samples = get_num_samples(dirname)

	if feature == None:

		# Loads all the units from all the documents
		all_units = LoadAllUnitsFromFiles(dirname, funit=funit)

		feature = DefineFeature(RemoveItemsWithPOSExcept(all_units), frep=frep) 

		# Rearrange the feature so that it can be used for broadcasting
		feature = feature.reshape([feature.shape[0], 1])

	# Init the features matrix, uint16 to save space.
	features = np.zeros((num_samples, feature.shape[0]), dtype=DataType(ftype))

	# Counter
	count = 0;

	# Clean up memory, hopefully python garbage collects
	all_units = 0;

	# Just to keep track on what has been done
	sys.stdout.write("\nDocuments Processed: 00000")

	# iterate over all the files in the directory
	for filename in os.listdir(dirname):
		if '.srl' in filename:
			features[count, :] = ExtractFeature(feature, DefineFeature(ReadDependencyParseFile(os.path.join(dirname, filename), funit=funit), frep=frep) ,ftype=ftype)
			count = count + 1
			sys.stdout.write("\b\b\b\b\b" + str(count).zfill(5)) # print just to see code is progressing

	if ftype == FeatureType.TFIDF:
		# Augmented Term Frequency
		# TF = K * binaryTF + (1-K) * count/(maxcount)
		# IDF = log (NumDocuments / Number of Documents which the term appears in)
		TF = (K * (features > 0)) + ((1-K) *  features) / (1 + DataType(FeatureType.TFIDF)(np.max(features, axis=1))).reshape([features.shape[0], 1])
		IDF = np.log(num_samples / (1 + DataType(FeatureType.TFIDF)(np.sum(features > 0, axis = 0))))
		features = TF * IDF
		TF = 0
		IDF = 0

	features = csr_matrix(features, dtype=DataType(ftype))
	return (feature, features) 

def get_num_samples(dirname):
	count = 0
	for filename in os.listdir(dirname):
		if '.srl' in filename:
			count = count + 1
	return count

def RemoveItemsWithPOSExcept(words_or_deps, keepers=None):
	"""
		For each word object in words_or_deps, removes it if the word's POS tag is not in keepers
		For each dependency object in words_or_deps, removes it if both the head and complement are not in keepers
		Returns the resulting list
	"""
	if keepers == None:
		keepers = KeeperPOS()

	result = []
	for w_or_d in words_or_deps:
		if w_or_d.__class__ == Word:
			if w_or_d.posTag in keepers:
				result.append(w_or_d)
		elif w_or_d.__class__ == Dependency:
			if w_or_d.head.posTag in keepers and w_or_d.complement.posTag in keepers:
				result.append(w_or_d)
	return result

def DefineFeature(words_or_deps, frep=FeatureRepresentation.HASH):
	"""
		Convert a list of Dependency objects into an numpy array of strings
	"""
	f = ConversionFunction(frep)
	return np.array([f(w_or_d) for w_or_d in set(words_or_deps)], dtype=DataType(frep)) 

def NumberOfHashCollisions(words_or_deps):
	return len([hash(w_or_d) for w_or_d in set(words_or_deps)]) - len(set([hash(w_or_d) for w_or_d in set(words_or_deps)])) 

def ExtractFeature(ffv, allff, ftype=FeatureType.BINARY):
	"""
		Inputs: ffv - the feature definition by FastFeatures or FastFeaturesHash, a 1 by N numpy array
				allff - the fast features of a given document, a 1 by M numpy array

		Output: a 1 by N vector, s.t. the i^th element of the output 
					is number of times the i^th element of ffv appeared in allff
	"""	
	# Transpose so dimensions match up and then sum, gives the count of how many times each element of ffv appears
	if ftype == FeatureType.BINARY:
		return np.any(ffv == allff, axis=1)
	else:
		return np.sum(ffv == allff, axis=1, dtype=DataType(FeatureType.COUNT)) # Make sure this is ok interms of MAXing out values
			 
def KeeperPOS():
	return ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RR", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

def LoadAllUnitsFromFiles(dirname, funit=FeatureUnits.WORD):
	deps = []
	count = 0
	sys.stdout.write("\nDocuments Processed: 00000")
	for filename in os.listdir(dirname):
		if '.srl' in filename:
			count = count + 1
			sys.stdout.write("\b\b\b\b\b" + str(count).zfill(5))
			deps = deps + ReadDependencyParseFile(os.path.join(dirname, filename), funit=funit)
	sys.stdout.write('\n')
	return deps
	
def Display(dep):
	"""
		Prints each dependency out on a seperate line
	"""
	for d in dep:
		if d.__class__ == Word:
			print(d.form)
		elif d.__class__ == Dependency:
			print(d.fancy_string())


