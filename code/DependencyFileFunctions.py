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
		return self.words == other.words
	def __ne__(self, other):
		return not self.eq(other)

	def __str__(self):
		s = "{" + str(self.head)+ ", " + str(self.complement) +  "} -- Sentence #" + str(self.sentenceNo) 
		return s
		


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

	
def Display(dep):
	"""
		Prints each dependency out on a seperate line
	"""
	dep = sorted(dep, key=lambda w:w.sentenceNo)
	for d in dep:
		print str(d)


