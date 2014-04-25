import numpy as np
import os
import tempfile
import sys
#from scipy.sparse import csr_matrix
from Util import *
from enum import Enum
import re
from copy import copy
import itertools

global USE_LEMMA 
USE_LEMMA = True

global CASE_SENSITIVE
CASE_SENSITIVE = True

global USE_DEP_TAGS
USE_DEP_TAGS = False

global USE_POS_TAGS
USE_POS_TAGS = False

global USE_ARG_LABELS
USE_ARG_LABELS = False

global SYMBOLS_TO_REMOVE
SYMBOLS_TO_REMOVE = '[!@#\$%\^&\*\(\)-_=\+\[\]\{\}\\\|;:\'\",\.<>/\?`~]'

global SYMBOLS_TO_KEEP
SYMBOLS_TO_KEEP = '[a-zA-Z0-9]*'

global REMOVE_SINGLE_CHARACTERS
REMOVE_SINGLE_CHARACTERS = True

global REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT
REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = True

global REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME
REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = True

global USE_MEMORY_MAP
USE_MEMORY_MAP = False

global KEEPER_POS
KEEPER_POS = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RR", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]



def DisplayConfiguration():
	print("Feature Configuration Settings")
	print("------------------------------")
	print("USE_LEMMA: " + str(USE_LEMMA))
	print("CASE_SENSITIVE: " + str(CASE_SENSITIVE))
	print("USE_POS_TAGS" + str(USE_POS_TAGS))
	print("USE_DEP_TAGS" + str(USE_DEP_TAGS))
	print("USE_ARG_LABELS" + str(USE_ARG_LABELS))
	print("SYMBOLS_TO_KEEP: " + SYMBOLS_TO_KEEP)
	print("REMOVE_SINGLE_CHARACTERS: " + str(REMOVE_SINGLE_CHARACTERS))
	print("REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT: " + str(REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT))
	print("REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME: " + str(REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME))
	print("USE_MEMORY_MAP: " + str(USE_MEMORY_MAP))
	print("KEEPER_POS: " + str(KEEPER_POS))


def DataType(argin):
		if argin == FeatureType.BINARY:
			return np.bool
		elif argin == FeatureType.TFIDF:
			return np.float16
		elif argin == FeatureType.COUNT:
			return np.dtype(int)
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
	PRED_ARG = "PA"
	WORDS_PA = "WPA"

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
		if USE_DEP_TAGS:
			if USE_POS_TAGS:
				if USE_LEMMA:
					return self.lemma + "/" + self.posTag + "-" + self.depRel
				else:
					if CASE_SENSITIVE:
						return self.form + "/" + self.posTag + "-" + self.depRel
					else:
						return self.form.lower() + "/" + self.posTag + "-" + self.depRel
			else:
				if USE_LEMMA:
					return self.lemma  + "-" + self.depRel
				else:
					if CASE_SENSITIVE:
						return self.form  + "-" + self.depRel
					else:
						return self.form.lower() + "-" + self.depRel
		else:
			if USE_POS_TAGS:
				if USE_LEMMA:
					return self.lemma + "/" + self.posTag 
				else:
					if CASE_SENSITIVE:
						return self.form + "/" + self.posTag 
					else:
						return self.form.lower() + "/" + self.posTag
			else:
				if USE_LEMMA:
					return self.lemma  
				else:
					if CASE_SENSITIVE:
						return self.form 
					else:
						return self.form.lower()

	def __hash__(self):
		return hash(str(self))

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
		return str(self.head) + " " + str(self.complement)

	def __hash__(self):
		return hash(str(self))

class PredicateArgument:
	"""
		A Structure for predicate argument structure
	"""

	def __init__(self, pred, args):
		self.pred = pred
		self.args = args
		
	def __str__(self):
		if not self.args:
			return "{Pred: " + (self.pred.form) + "}"
		s = "{Pred: " + (self.pred.form)
		arg_labels = self.args.keys()
		arg_labels.sort()
		s = s + "| " + arg_labels[0] + ":"
		for w in self.args[arg_labels[0]]:
			s = s + " " + (w.form)
		for a in range(1, len(arg_labels)):
			s = s + ", " + arg_labels[a] + ": " 
			for w in self.args[arg_labels[a]]:
				s = s + " " + (w.form)
		s = s + "}"
		return s

	def getFeatures(self, frep=FeatureRepresentation.HASH):
		f_def = list()
		f = ConversionFunction(frep)
		f_def.append(f(self.pred))
		for a_label in self.args:
				# Convert list of words into string
				word_string = str(self.args[a_label][0])
				for w in range(1, len(self.args[a_label])):
					word_string = word_string + " " + str(self.args[a_label][w])
				if USE_ARG_LABELS:
					f_def.append(f(a_label + " " + word_string))
				else:
					f_def.append(f(word_string))
		return f_def

	def __hash__(self):
		return hash(str(self))
	

def Features(dirname, funit=FeatureUnits.WORD, ftype=FeatureType.BINARY, frep=FeatureRepresentation.HASH, feature=None, K=0.5, UseLemma=True):
	"""
		Creates an M-by-N matrix where N is the length of the feature vector and M is number of documents
		The documents used are all the .srl files stored in the directory dirname
	"""		

	USE_LEMMA = UseLemma

	f_data_type = DataType(ftype);

	num_samples = get_num_samples(dirname)

	is_testing = not (feature == None)

	if feature == None:

		# Loads all the units from all the documents
		all_units = LoadAllUnitsFromFiles(dirname, funit=funit, keep_duplicates=False, remove_stop_words=True)

		feature = DefineFeature(all_units, frep=frep, funit=funit) 

		# Rearrange the feature so that it can be used for broadcasting
		feature = feature.reshape([feature.shape[0], 1])

	# Init the features matrix, uint16 to save space.
	if USE_MEMORY_MAP:
		if is_testing:
			features = np.memmap("testing_features.dat", shape=(num_samples, feature.shape[0]), dtype=DataType(ftype), mode='w+')
		else:
			features = np.memmap("training_features.dat", shape=(num_samples, feature.shape[0]), dtype=DataType(ftype), mode='w+')
	else:
		features = np.zeros((num_samples, feature.shape[0]), dtype=DataType(ftype))


	# Counter
	count = 0;

	# Clean up memory, hopefully python garbage collects
	all_units = 0;

	# Just to keep track on what has been done
	sys.stdout.write("\nExtracting Features, Documents Processed: 00000")

	# iterate over all the files in the directory
	for filename in os.listdir(dirname):
		if '.srl' in filename:
			features[count, :] = ExtractFeature(feature, Convert((ReadDependencyParseFile(os.path.join(dirname, filename), remove=True, funit=funit)),frep=frep, funit=funit),ftype=ftype)
			count = count + 1
			sys.stdout.write("\b\b\b\b\b" + str(count).zfill(5)) # print just to see code is progressing

	# Feature Reduction
	if not is_testing:
		if REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME:
			num_occurances = np.sum(features, axis=0)
			feature = feature[ num_occurances > 1 ]
			features = features[:, num_occurances > 1]

		if REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT:
			num_occurances = np.sum(features > 0, axis=0)
			feature = feature[ num_occurances > 1 ]
			features = features[:, num_occurances > 1]

	if ftype == FeatureType.TFIDF:
		# Augmented Term Frequency
		# TF = K * binaryTF + (1-K) * count/(maxcount)
		# IDF = log (NumDocuments / Number of Documents which the term appears in)
		TF = (K * (features > 0)) + ((1-K) *  features) / (1 + DataType(FeatureType.TFIDF)(np.max(features, axis=1))).reshape([features.shape[0], 1])
		IDF = np.log(num_samples / (1 + DataType(FeatureType.TFIDF)(np.sum(features > 0, axis = 0))))
		features = TF * IDF
		TF = 0
		IDF = 0
	elif ftype == FeatureType.BINARY:
		features = features > 0


	#features = csr_matrix(features, dtype=DataType(ftype))
	if is_testing:
		return features
	else:
		return (feature, features) 


def ToTFIDF(features, K=0.5):
	num_samples = features.shape[0]
	TF = (K * (features > 0)) + ((1-K) *  features) / (1 + DataType(FeatureType.TFIDF)(np.max(features, axis=1))).reshape([features.shape[0], 1])
	IDF = np.log(num_samples / (1 + DataType(FeatureType.TFIDF)(np.sum(features > 0, axis = 0))))
	return TF * IDF

def ToBINARY(features):
	return features > 0

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
				# Remove Symbols
				if len(re.sub(SYMBOLS_TO_REMOVE, '', str(w_or_d))) == len(str(w_or_d)):
					if (not REMOVE_SINGLE_CHARACTERS) or len(str(w_or_d)) > 1:
						result.append(w_or_d)
		elif w_or_d.__class__ == Dependency:
			if w_or_d.head.posTag in keepers and w_or_d.complement.posTag in keepers:
				if len(re.sub(SYMBOLS_TO_REMOVE, '', str(w_or_d.head))) == len(str(w_or_d.head)):
					if len(re.sub(SYMBOLS_TO_REMOVE, '', str(w_or_d.complement))) == len(str(w_or_d.complement)):
						if (not REMOVE_SINGLE_CHARACTERS) or len(str(w_or_d.head)) > 1:
							if (not REMOVE_SINGLE_CHARACTERS) or len(str(w_or_d.head)) > 1:
								result.append(w_or_d)
		elif w_or_d.__class__ == PredicateArgument:
			if w_or_d.predicate.posTag in keepers:
				if np.all(np.array([ (w.posTag in keepers) for w in w_or_d.args.values()])):
					result.append()
	return result

def Convert(units, frep=FeatureRepresentation.HASH, funit=FeatureUnits.WORD):
	"""
		Convert a list of Dependency objects into an numpy array of strings
	"""
	if funit == FeatureUnits.PRED_ARG:
		f_def = list()
		for u in units:
			f_def.extend(u.getFeatures(frep=frep))
		return np.array(f_def, dtype=DataType(frep))
	else:
		conv = ConversionFunction(frep)
		return np.array([conv(u) for u in units], dtype=DataType(frep)) 

def ConvertUnit(words_or_deps, frep=FeatureRepresentation.HASH):
	"""
		Convert a list of Dependency objects into an numpy array of strings
	"""
	f = ConversionFunction(frep)
	return f(words_or_deps) 

def DefineFeature(units, frep=FeatureRepresentation.HASH, funit=FeatureUnits.WORD):
	"""
		Convert a list of Dependency objects into an numpy array of strings
	"""
	if funit == FeatureUnits.PRED_ARG:
		f_def = list()
		for pa in units:
			f_def.extend(pa.getFeatures(frep=frep))
		return np.array(list(set(f_def)), dtype=DataType(frep))
	else:
		conv = ConversionFunction(frep)
		return np.array([conv(u) for u in set(units)], dtype=DataType(frep)) 

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
	# if ftype == FeatureType.BINARY:
	# 	return np.any(ffv == allff, axis=1)
	# else:
	return np.sum(ffv == allff, axis=1, dtype=DataType(FeatureType.COUNT)) # Make sure this is ok interms of MAXing out values
			 
def KeeperPOS():
	return KEEPER_POS

def LoadAllUnitsFromFiles(dirname, funit=FeatureUnits.WORD, keep_duplicates=False, remove_stop_words=True):
	deps = []
	count = 0
	sys.stdout.write("\nDocuments Processed: 00000")
	for filename in os.listdir(dirname):
		if '.srl' in filename:
			count = count + 1
			sys.stdout.write("\b\b\b\b\b" + str(count).zfill(5))
			if keep_duplicates:
				deps.extend((ReadDependencyParseFile(os.path.join(dirname, filename), remove=remove_stop_words, funit=funit)))
			else:
				deps.extend(set(ReadDependencyParseFile(os.path.join(dirname, filename), remove=remove_stop_words, funit=funit)))
	sys.stdout.write('\n')
	return deps
	
def Display(dep):
	"""
		Prints each dependency out on a seperate line
	"""
	for d in dep:
		if d.__class__ == Word:
			print(str(d))
		elif d.__class__ == Dependency:
			print(str(d))
		elif d.__class__ == PredicateArgument:
			print(str(d))


def ReadDependencyParseFile(filename, funit=FeatureUnits.WORD, remove=True):
	""" 
		This function reads a dep format file into a (python) list of Word and or Dependency objects.
	"""
	Words = list()
	Dependencies = list()
	PredArgs = list()

	keepers = KeeperPOS()

	if funit == FeatureUnits.WORD or funit == FeatureUnits.BOTH or funit == FeatureUnits.WORDS_PA:
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
				if (not remove) or (posTag in keepers and ((USE_LEMMA and not_single_character(lemma)) or ((not USE_LEMMA) and not_single_character(wordform)))  and ((USE_LEMMA and not_contains_symbols(lemma)) or ((not USE_LEMMA) and not_contains_symbols(wordform)))):
					Words.append(Word(wordform, lemma, posTag, feat, depRel))
				# else:
				# 	print([wordno, wordform, lemma, posTag, feat, head, depRel])
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
				if not head == -1:
					if (sentenceNo, head) in ComplementsOfHeadInSentence:
						if (not remove) or (posTag in keepers and ((USE_LEMMA and not_single_character(lemma)) or ((not USE_LEMMA) and not_single_character(wordform)))  and ((USE_LEMMA and not_contains_symbols(lemma)) or ((not USE_LEMMA) and not_contains_symbols(wordform)))):
							ComplementsOfHeadInSentence[(sentenceNo, head)].append(wordno)
					else:
						if (not remove) or (posTag in keepers and ((USE_LEMMA and not_single_character(lemma)) or ((not USE_LEMMA) and not_single_character(wordform)))  and ((USE_LEMMA and not_contains_symbols(lemma)) or ((not USE_LEMMA) and not_contains_symbols(wordform)))):
							ComplementsOfHeadInSentence[(sentenceNo, head)] = [wordno]
			else:
				sentenceNo = sentenceNo + 1
				WordsInSentence[sentenceNo] = []
		for (sentno, headno) in ComplementsOfHeadInSentence:
			for compno in ComplementsOfHeadInSentence[(sentno, headno)]:
				if (not remove) or (WordsInSentence[sentno][headno].posTag in keepers and ((USE_LEMMA and not_single_character(WordsInSentence[sentno][headno].lemma)) or ((not USE_LEMMA) and not_single_character(WordsInSentence[sentno][headno].form)))  and ((USE_LEMMA and not_contains_symbols(WordsInSentence[sentno][headno].lemma)) or ((not USE_LEMMA) and not_contains_symbols(WordsInSentence[sentno][headno].form)))):
					Dependencies.append(Dependency(WordsInSentence[sentno][headno], WordsInSentence[sentno][compno], sentno))
		
		if funit == FeatureUnits.DEPENDENCY_PAIR:
			return Dependencies
		elif funit == FeatureUnits.BOTH:
			return Dependencies + Words

	if funit == FeatureUnits.PRED_ARG or funit == FeatureUnits.WORDS_PA:
		f = open(filename, 'r')
		# Mapping from (sentence number, head id number) to a list of the complements of the head id number
		ComplementsOfHeadInSentence = dict()
		WordsInSentence = dict()
		WordsInSentence[0] = []
		sentenceNo = 0
		# Mapping from (sentence number, rel id number) to a list of dictionaries of (arglabel, word id number)
		PredArgInSentence = dict()
		for line in f:
			spl = line.split()
			if len(spl) == 8:
				wordno = int(spl[0])-1
				wordform = spl[1]
				lemma = spl[2]
				posTag = spl[3]
				feat = spl[4]
				head = int(spl[5])-1
				depRel = spl[6]
				argTag = spl[7]
				WordsInSentence[sentenceNo].append(Word(wordform, lemma, posTag, feat, depRel))
				if not argTag=='_':
					args = argTag.split(";")
					for a in args:
						tuple_rel_arglabel = a.split(":")
						rel = int(tuple_rel_arglabel[0])-1
						arglabel = tuple_rel_arglabel[1]
						if (sentenceNo, rel) in PredArgInSentence:
							#if (not remove) or (posTag in keepers and ((USE_LEMMA and not_single_character(lemma)) or ((not USE_LEMMA) and not_single_character(wordform)))  and ((USE_LEMMA and not_contains_symbols(lemma)) or ((not USE_LEMMA) and not_contains_symbols(wordform)))):
							PredArgInSentence[(sentenceNo, rel)].append((arglabel,wordno))
						else:
							#if (not remove) or (posTag in keepers and ((USE_LEMMA and not_single_character(lemma)) or ((not USE_LEMMA) and not_single_character(wordform)))  and ((USE_LEMMA and not_contains_symbols(lemma)) or ((not USE_LEMMA) and not_contains_symbols(wordform)))):
							PredArgInSentence[(sentenceNo, rel)] = [(arglabel,wordno)]
				if not head == -1:
					if (sentenceNo, head) in ComplementsOfHeadInSentence:
						#if (not remove) or (posTag in keepers and ((USE_LEMMA and not_single_character(lemma)) or ((not USE_LEMMA) and not_single_character(wordform)))  and ((USE_LEMMA and not_contains_symbols(lemma)) or ((not USE_LEMMA) and not_contains_symbols(wordform)))):
						ComplementsOfHeadInSentence[(sentenceNo, head)].append(wordno)
					else:
						#if (not remove) or (posTag in keepers and ((USE_LEMMA and not_single_character(lemma)) or ((not USE_LEMMA) and not_single_character(wordform)))  and ((USE_LEMMA and not_contains_symbols(lemma)) or ((not USE_LEMMA) and not_contains_symbols(wordform)))):
						ComplementsOfHeadInSentence[(sentenceNo, head)] = [wordno]
			else:
				sentenceNo = sentenceNo + 1
				WordsInSentence[sentenceNo] = []
		for (sentno, relno) in PredArgInSentence:
			if (not remove) or (WordsInSentence[sentno][relno].posTag in keepers and ((USE_LEMMA and not_single_character(WordsInSentence[sentno][relno].lemma)) or ((not USE_LEMMA) and not_single_character(WordsInSentence[sentno][relno].form)))  and ((USE_LEMMA and not_contains_symbols(WordsInSentence[sentno][relno].lemma)) or ((not USE_LEMMA) and not_contains_symbols(WordsInSentence[sentno][relno].form)))):
				args = PredArgInSentence[(sentno, relno)]
				args_with_word_objects = dict()
				for a in args:
					list_of_word_nos = list()
					if (not remove) or (WordsInSentence[sentno][a[1]].posTag in keepers and ((USE_LEMMA and not_single_character(WordsInSentence[sentno][a[1]].lemma)) or ((not USE_LEMMA) and not_single_character(WordsInSentence[sentno][a[1]].form)))  and ((USE_LEMMA and not_contains_symbols(WordsInSentence[sentno][a[1]].lemma)) or ((not USE_LEMMA) and not_contains_symbols(WordsInSentence[sentno][a[1]].form)))):
						list_of_word_nos.extend([a[1]])
					if (sentno, a[1]) in ComplementsOfHeadInSentence:
						for wn in ComplementsOfHeadInSentence[(sentno, a[1])]:
							if (not remove) or (WordsInSentence[sentno][wn].posTag in keepers and ((USE_LEMMA and not_single_character(WordsInSentence[sentno][wn].lemma)) or ((not USE_LEMMA) and not_single_character(WordsInSentence[sentno][wn].form)))  and ((USE_LEMMA and not_contains_symbols(WordsInSentence[sentno][wn].lemma)) or ((not USE_LEMMA) and not_contains_symbols(WordsInSentence[sentno][wn].form)))):
								list_of_word_nos.extend([wn]);
					list_of_word_nos = list(set(list_of_word_nos))
					list_of_word_nos.sort()
					list_of_word_obj = [WordsInSentence[sentno][wno] for wno in list_of_word_nos]
					if len(list_of_word_obj) > 0:
						args_with_word_objects[a[0]] = copy(list_of_word_obj);
				PredArgs.append(PredicateArgument(WordsInSentence[sentno][relno], copy(args_with_word_objects)))
		if funit == FeatureUnits.PRED_ARG:
			return PredArgs
		elif funit == FeatureUnits.WORDS_PA:
			return Words + PredArgs

def not_contains_symbols(s):
	return len(re.sub(SYMBOLS_TO_KEEP, '', str(s))) == 0

def not_single_character(s):
	return len(s) > 1









