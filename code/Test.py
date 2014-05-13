import Features
import Word2VecExecuter
import sys
import random

def runAllTests(): 
#	test001Word2VecHash()
	test002LookForDuplicateHash()
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
	del model 

	#check if after deleting the model it still works
	word = dictWord2Vec.keys()[random_word_index]
	pointer = dictWord2Vec[word]

	vec1ad = fpr[pointer+3]
	assert (len(vec1ad)==len(vec1))

def test002LookForDuplicateHash():
	
	Features.USE_LEMMA = False
	Features.CASE_SENSITIVE = True
#	Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
#	Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
#	Features.DisplayConfiguration()
	
	
#	(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.WORD,frep=Features.FeatureRepresentation.STRING)
#	(fdef2, X2) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.WORD,frep=Features.FeatureRepresentation.HASH)
	
	# Loads all the units from all the documents
#	all_units = Features.LoadAllUnitsFromFiles('../data_sets/small/train/', funit=Features.FeatureUnits.WORD, keep_duplicates=False, remove_stop_words=False)
	all_units = Features.LoadAllUnitsFromFiles('../data_sets/reuters_21578/train copy/', funit=Features.FeatureUnits.WORD, keep_duplicates=False, remove_stop_words=False)

	feature_hash = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.HASH, funit=Features.FeatureUnits.WORD) 
	feature_string = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.STRING, funit=Features.FeatureUnits.WORD) 

	# Rearrange the feature_string so that it can be used for broadcasting
	#feature_hash = feature_hash.reshape([feature_hash.shape[0], 1])	
	#feature_string = feature_string.reshape([feature_string.shape[0], 1])	
	print("\nFeature String Definition:")
	print(feature_string)
	print("\nFeature Hash Definition:")
	print(feature_hash)

	set_string = set(feature_string)
	set_hash = set(feature_hash) 
	
	print("length of string set: " + str(len(set_string)))
	print("length of hash set: " + str(len(set_hash)))
	
	assert(len(set_string)==len(set_hash))
	
	
	
runAllTests()