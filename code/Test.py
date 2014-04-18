import Features
import Word2VecExecuter
import sys
import random

def runAllTests(): 
	test001Word2VecHash()
	sys.stdout.write("\nTESTS FINISHED SUCCESSFULLY!!!")

def test001Word2VecHash():
	"""Tests the hashing of all the words in the documents in a particular directory"""

	#model = Word2VecExecuter.Word2VecGetModel("../data_sets/GoogleNews-vectors-negative300.bin")
	model = Word2VecExecuter.Word2VecGetModel("tools/word2vec/word2vec-read-only/vectors.bin")
	list_of_words = Features.LoadAllUnitsFromFiles("../data_sets/reuters_21578/test_train/")
	dictWord2Vec, fpr = Word2VecExecuter.Word2VecLoadWordsHashTable(model, list_of_words)

	assert len(dictWord2Vec)==len(fpr)

	random_word_index = random.randint(0,len(fpr)-1)
	sys.stdout.write("\nrandom tested word index: " + str(random_word_index) + " " + str(len(dictWord2Vec)) + " " + str(len(fpr)))
	word = dictWord2Vec.keys()[random_word_index]
	pointer = dictWord2Vec[word]

	vec1 = fpr[pointer]
	vec2 = Word2VecExecuter.Word2VecGetVector(model,word)
	comp_vec = vec1-vec2

	assert max(abs(comp_vec))==0.0

	dictWord2VecH, fprH = Word2VecExecuter.Word2VecLoadWordsHashTable(model, list_of_words, Features.FeatureRepresentation.HASH)

	hashed_word = dictWord2VecH.keys()[random_word_index]
	assert word != hashed_word
	pointerH = dictWord2VecH[hashed_word]
	vec3H = fpr[pointerH]
	comp_vec = vec1-vec3H

	assert max(abs(comp_vec))==0.0

