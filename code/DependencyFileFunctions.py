import numpy as np


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


class Dependency:
	"""
		A Dependency object consists of a list of words 
		and the sentence that the dependency appears in
		the sentence numbers start at 0.

		Two dependencies are equal if their word sets are equal
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
		


def ReadDependencyParseFile(filename):
	""" 
		This function reads a dep format file into a list of Dependency objects.
	"""

	f = open(filename, 'r')
	ComplementsOfHeadInSentence = dict()
	WordsInSentence = dict()
	WordsInSentence[0] = []
	sentenceNo = 0
	for line in f:
		spl = line.split()
		if len(spl) == 7:
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


def ExtractFeature(feature, dep_document):
	feat = np.zeros_like(feature)
	for dep in dep_document:
		try: 
			feat[feature.index(dep)] = feat[feature.index(dep)] + 1
		except:
			None
	return feat
		 

def DefineFeature(list_of_all_deps_in_all_files):
	feature = []
	keepers = KeeperPOS()
	for dep in list_of_all_deps_in_all_files:
		if dep.head.posTag in keepers and dep.complement.posTag in keepers:
			feature.append(dep)
	return list(set(feature))



def KeeperPOS():
	return ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RR", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]


	
def Display(dep):
	"""
		Prints each dependency out on a seperate line
	"""
	dep = sorted(dep, key=lambda w:w.sentenceNo)
	for d in dep:
		print str(d)


