import Features
import Word2VecExecuter
import sys
import random

def runAllTests(): 
	test001Word2VecHash()
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
	'''Checks if there are collisions among hashes'''
		
	Features.USE_LEMMA = False
	Features.CASE_SENSITIVE = True
	parsedFilesDirectory = '../data_sets/reuters_21578/train copy/'
	###########################################FOR WORDS########################################
	all_units = Features.LoadAllUnitsFromFiles(parsedFilesDirectory, funit=Features.FeatureUnits.WORD, keep_duplicates=False, remove_stop_words=False)

	feature_hash = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.HASH, funit=Features.FeatureUnits.WORD) 
	feature_string = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.STRING, funit=Features.FeatureUnits.WORD) 


	set_string = set(feature_string)
	set_hash = set(feature_hash) 
	
	print("length of string set (words): " + str(len(set_string)))
	print("length of hash set (words): " + str(len(set_hash)))
	
	assert(len(set_string)==len(set_hash))
	
	########################################FOR DEPENDENCY PAIRS########################################
	all_units = Features.LoadAllUnitsFromFiles(parsedFilesDirectory, funit=Features.FeatureUnits.DEPENDENCY_PAIR, keep_duplicates=False, remove_stop_words=False)
	
	feature_hash = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.HASH, funit=Features.FeatureUnits.DEPENDENCY_PAIR) 
	feature_string = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.STRING, funit=Features.FeatureUnits.DEPENDENCY_PAIR) 
	
	print("\nFeature String Definition:")
	print(feature_string)
	print("\nFeature Hash Definition:")
	print(feature_hash)
	
	set_string = set(feature_string)
	set_hash = set(feature_hash) 
	
	print("length of string set (dependency pairs): " + str(len(set_string)))
	print("length of hash set (dependency pairs): " + str(len(set_hash)))
	
	assert(len(set_string)==len(set_hash))	
	
	########################################FOR WORDS AND DEPENDENCY PAIRS########################################
	
	all_units = Features.LoadAllUnitsFromFiles(parsedFilesDirectory, funit=Features.FeatureUnits.WORDS_AND_DEPENDENCY_PAIRS, keep_duplicates=False, remove_stop_words=False)
	
	feature_hash = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.HASH, funit=Features.FeatureUnits.WORDS_AND_DEPENDENCY_PAIRS) 
	feature_string = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.STRING, funit=Features.FeatureUnits.WORDS_AND_DEPENDENCY_PAIRS) 
	
	print("\nFeature String Definition:")
	print(feature_string)
	print("\nFeature Hash Definition:")
	print(feature_hash)
	
	set_string = set(feature_string)
	set_hash = set(feature_hash) 
	
	print("length of string set (words and dependency pairs): " + str(len(set_string)))
	print("length of hash set (words and dependency pairs): " + str(len(set_hash)))
	
	assert(len(set_string)==len(set_hash))		
	
	
	########################################FOR PREDICATE ARGUMENT########################################
	
	all_units = Features.LoadAllUnitsFromFiles(parsedFilesDirectory, funit=Features.FeatureUnits.PREDICATE_ARGUMENT, keep_duplicates=False, remove_stop_words=False)
	
	feature_hash = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.HASH, funit=Features.FeatureUnits.PREDICATE_ARGUMENT) 
	feature_string = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.STRING, funit=Features.FeatureUnits.PREDICATE_ARGUMENT) 
	
	print("\nFeature String Definition:")
	print(feature_string)
	print("\nFeature Hash Definition:")
	print(feature_hash)
	
	set_string = set(feature_string)
	set_hash = set(feature_hash) 
	
	print("length of string set (predicate argument): " + str(len(set_string)))
	print("length of hash set (predicate argument): " + str(len(set_hash)))
	
	assert(len(set_string)==len(set_hash))	

	########################################FOR WORDS AND PREDICATE ARGUMENT########################################
	
	all_units = Features.LoadAllUnitsFromFiles(parsedFilesDirectory, funit=Features.FeatureUnits.WORDS_AND_PREDICATE_ARGUMENT, keep_duplicates=False, remove_stop_words=False)
	
	feature_hash = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.HASH, funit=Features.FeatureUnits.WORDS_AND_PREDICATE_ARGUMENT) 
	feature_string = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.STRING, funit=Features.FeatureUnits.WORDS_AND_PREDICATE_ARGUMENT) 
	
	print("\nFeature String Definition:")
	print(feature_string)
	print("\nFeature Hash Definition:")
	print(feature_hash)
	
	set_string = set(feature_string)
	set_hash = set(feature_hash)
	
	print("length of string set (words and predicate argument): " + str(len(set_string)))
	print("length of hash set (words and predicate argument): " + str(len(set_hash)))
	
	assert(len(set_string)==len(set_hash))
	
	########################################FOR DEPENDENCY PAIRS AND PREDICATE ARGUMENT########################################
	all_units = Features.LoadAllUnitsFromFiles(parsedFilesDirectory, funit=Features.FeatureUnits.DEPENDENCY_PAIRS_AND_PREDICATE_ARGUMENT, keep_duplicates=False, remove_stop_words=False)
	
	feature_hash = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.HASH, funit=Features.FeatureUnits.DEPENDENCY_PAIRS_AND_PREDICATE_ARGUMENT) 
	feature_string = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.STRING, funit=Features.FeatureUnits.DEPENDENCY_PAIRS_AND_PREDICATE_ARGUMENT) 
	
	print("\nFeature String Definition:")
	print(feature_string)
	print("\nFeature Hash Definition:")
	print(feature_hash)
	
	set_string = set(feature_string)
	set_hash = set(feature_hash)
	
	print("length of string set (dependency pairs and predicate argument): " + str(len(set_string)))
	print("length of hash set (dependency pairs and predicate argument): " + str(len(set_hash)))
	
	assert(len(set_string)==len(set_hash))		
	
	########################################ALL########################################
	all_units = Features.LoadAllUnitsFromFiles(parsedFilesDirectory, funit=Features.FeatureUnits.ALL, keep_duplicates=False, remove_stop_words=False)
	
	feature_hash = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.HASH, funit=Features.FeatureUnits.ALL) 
	feature_string = Features.DefineFeature(all_units, frep=Features.FeatureRepresentation.STRING, funit=Features.FeatureUnits.ALL) 
	
	print("\nFeature String Definition:")
	print(feature_string)
	print("\nFeature Hash Definition:")
	print(feature_hash)
	
	set_string = set(feature_string)
	set_hash = set(feature_hash)
	
	print("length of string set (all): " + str(len(set_string)))
	print("length of hash set (all): " + str(len(set_hash)))
	
	assert(len(set_string)==len(set_hash))				

	
runAllTests()