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
	def __init__(self, words, sentenceNo):
		self.words = words
		self.sentenceNo = sentenceNo

	def __eq__(self, other):
		return self.words == other.words
	def __ne__(self, other):
		return not self.eq(other)

	def __str__(self):
		s = "{" + str(self.words[0]) 
		for i in range(1, len(self.words)):
			s = s + ", " + str(self.words[i])
		s = s + "} -- Sentence #" + str(self.sentenceNo) 
		return s
		


def ReadDependencyParseFile(filename):
	""" 
		This function reads a dep format file into a list of Dependency objects.
	"""

	f = open(filename, 'r')
	tmp = dict()
	sentenceNo = 0
	for line in f:
		spl = line.split()
		if len(spl) == 7:
			if (sentenceNo, spl[5]) in tmp:
				tmp[(sentenceNo, spl[5])].append(Word(spl[1], spl[2], spl[3], spl[4], spl[6]))
			else:
				tmp[(sentenceNo, spl[5])] = [(Word(spl[1], spl[2], spl[3], spl[4], spl[6]))]
		else:
			sentenceNo = sentenceNo + 1
	Dependencies = []
	for d in tmp:
		Dependencies.append(Dependency(tmp[d], d[0]))
	return Dependencies

	
def DisplayDependencies(dep):
	"""
		Prints each dependency out on a seperate line
	"""
	for d in dep:
		print str(d)
