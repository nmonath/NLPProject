class Word:
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
	for d in dep:
		print str(d)
