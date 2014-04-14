import numpy as np
import os
import tempfile
import sys
from scipy.sparse import csr_matrix
from Util import *

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
		return self.form

	def __hash__(self):
		return hash(self.lemma)

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

	def __str__(self):
		s = "{" + str(self.head)+ ", " + str(self.complement) +  "} -- Sentence #" + str(self.sentenceNo) 
		return s

	def __hash__(self):
		return (hash(self.head.lemma + self.complement.lemma))

def ReadWordsFromDependencyParseFile(filename):
	""" 
		This function reads a dep format file into a list of Dependency objects.
	"""

	f = open(filename, 'r')
	# Mapping from (sentence number, head id number) to a list of the complements of the head id number
	Words = list()
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
	return Words

def ReadDependencyParseFile(filename):
	""" 
		This function reads a dep format file into a list of Dependency objects.
	"""

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
	Dependencies = []
	for (sentno, headno) in ComplementsOfHeadInSentence:
		for compno in ComplementsOfHeadInSentence[(sentno, headno)]:
			#print str(sentno) + " " + str(headno) + " " + str(compno)
			#Display(WordsInSentence[sentno])
			Dependencies.append(Dependency(WordsInSentence[sentno][headno], WordsInSentence[sentno][compno], sentno))
	return Dependencies

def Features(dirname, method=('Dep', 'Hash'), feature=None):
	"""
		Creates an M-by-N matrix where N is the length of the feature vector and M is number of documents
		The documents used are all the .srl files stored in the directory dirname
	"""		

	num_samples = num_samples(dirname)

	if feature == None:

		# Loads all the dependencies from all the documents
		if method[0] == 'Dep':
			all_deps = LoadAllDepFromFiles(dirname)
		else:
			all_deps = LoadAllWordsFromFiles(dirname)



		# Defines a feature vector based on the dependencies
		feature_def = DefineFeature(all_deps, method=method[0])

		# Convert the list of Dependency objects to an numpy array of strings

		fast_feature_def = FastFeatures(feature_def, method=method[1])
		fast_feature_def = fast_feature_def.reshape([fast_feature_def.shape[0], 1])
		feature = fast_feature_def

	# Init the features matrix, uint16 to save space. We should maybe use a sparse matrix
	features = np.zeros((num_samples, len(feature)), dtype=np.uint16)

	# Counter
	count = 0;

	# Clean up memory, hopefully python garbage collects
	all_deps = 0;
	feature_def = 0;

	# Just to keep track on what has been done
	sys.stdout.write("\nDocuments Processed: 00000")

	# iterate over all the files in the directory
	for filename in os.listdir(dirname):
		if '.srl' in filename:
			sys.stdout.write("\b\b\b\b\b" + str(count).zfill(5)) # print just to see code is progressing
			# Extract the feature vector using the fast mechanism
			if method[0] == 'Dep':
				features[count, :] = ExtractFF(feature, FastFeatures(ReadDependencyParseFile(os.path.join(dirname, filename)), method=method[1]))
			else:
				features[count, :] = ExtractFF(feature, FastFeatures(ReadWordsFromDependencyParseFile(os.path.join(dirname, filename)), method=method[1]))
			count = count + 1
	return (feature, features) 

def num_samples(dirname):
	count = 0
	for filename in os.listdir(dirname):
		if '.srl' in filename:
			count = count + 1

def FastFeatures(deps, method='Hash'):
	"""
		Convert a list of Dependency objects into an numpy array of strings
	"""
	if method == 'NoHash':
		FF = np.empty((len(deps), ), dtype=np.object)
		count = 0;
		for d in deps:
			# Note the use of lemmatized version instead of the word form
			FF[count] = (d.head.lemma + " " + d.complement.lemma)
			count = count +1
		return FF
	elif method == 'Hash':
		FF = np.empty((len(deps), ), dtype=np.dtype(int))
		count = 0;
		for d in deps:
			# Note the use of the lemmatized version instead of the word fomr
			FF[count] = hash(d.head.lemma + " " + d.complement.lemma)
			count = count +1
		return FF
	elif method == 'WordNoHash':
		FF = np.empty((len(deps), ), dtype=np.object)
		count = 0;
		for f in deps:
			# Note the use of the lemmatized version instead of the word fomr
			FF[count] = (f.lemma)
			count = count +1
		return FF
	elif method == 'WordHash':
		FF = np.empty((len(deps), ), dtype=np.dtype(int))
		count = 0;
		for f in deps:
			# Note the use of the lemmatized version instead of the word fomr
			FF[count] = hash(f.lemma)
			count = count +1
		return FF

def ExtractFF(ffv, allff):
	"""
		Inputs: ffv - the feature definition by FastFeatures or FastFeaturesHash, a 1 by N numpy array
				allff - the fast features of a given document, a 1 by M numpy array

		Output: a 1 by N vector, s.t. the i^th element of the output 
					is number of times the i^th element of ffv appeared in allff
	"""

	
	# Transpose so dimensions match up and then sum, gives the count of how many times each element of ffv appears
	return np.sum(ffv == allff, axis=1, dtype=np.uint16)
	
def DefineFeature(list_of_all_deps_in_all_files, method='Dep'):
	"""
		Given all the Dependencies that appear in inputted list
		return a list containing only those Dependencies in which
		both the head and the complement have a POS tag in the KeeperPOS()
		list. 
	"""
	if method == 'Dep':
		feature = []
		keepers = KeeperPOS()
		for dep in list_of_all_deps_in_all_files:
			if dep.head.posTag in keepers and dep.complement.posTag in keepers:
				feature.append(dep)
		return list(set(feature))
	elif method == 'BOW':
		feature = []
		keepers = KeeperPOS()
		for w in list_of_all_deps_in_all_files:
			if w.posTag in keepers:
				feature.append(w)
		return list(set(feature))

def ExtractFeature(feature, dependencies_from_document):
	"""
		feature is a list of [(h1_f, c1_f), (h2_f, c2_f), ...] generated by DefineFeature
		dependencies_from_document is a list of dependencies that appear in a file, the list is read in by ReadDependencyParseFile

		Returns a vector v such that v[0] corresponds to (h1_f, c1_f), v[1] to (h2_f, c2_f), etc. 
		The value of v[i] is a count of the number of types (hi+1_f, ci+1_f) appears in dependencies_from_document
	"""
	feat = np.zeros_like(feature)
	for dep in dependencies_from_document:
		try: 
			feat[feature.index(dep)] = feat[feature.index(dep)] + 1
		except:
			None
	return feat
		 
def KeeperPOS():
	return ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RR", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

def LoadAllWordsFromFiles(dirname):
	deps = []
	count = 0
	sys.stdout.write("\nDocuments Processed: 00000")
	for filename in os.listdir(dirname):
		if '.srl' in filename:
			sys.stdout.write("\b\b\b\b\b" + str(count).zfill(5))
			count = count + 1
			deps = deps + ReadWordsFromDependencyParseFile(os.path.join(dirname, filename))
	sys.stdout.write('\n')
	return deps


def LoadAllDepFromFiles(dirname):
	deps = []
	count = 0
	sys.stdout.write("\nDocuments Processed: 00000")
	for filename in os.listdir(dirname):
		if '.srl' in filename:
			sys.stdout.write("\b\b\b\b\b" + str(count).zfill(5))
			count = count + 1
			deps = deps + ReadDependencyParseFile(os.path.join(dirname, filename))
	sys.stdout.write('\n')
	return deps
	
def Display(dep):
	"""
		Prints each dependency out on a seperate line
	"""
	dep = sorted(dep, key=lambda w:w.sentenceNo)
	for d in dep:
		print(str(d))

def LoadClassFile(filename):
	Y = list()
	f = open(filename, 'r')
	for line in f:
		spl = line.split(' ')
		Y.append(int(spl[0]))
	return np.array(Y, dtype=np.dtype(int))

