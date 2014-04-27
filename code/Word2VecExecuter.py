#working on it 
import subprocess
from gensim.models import word2vec
import numpy as np
import sys
from Features import ConvertUnit 
from Features import FeatureRepresentation
from Features import RemoveItemsWithPOSExcept

#check if use list, or one file, for now only using one file as in demo 
def Word2VecTrain(filenameOrig, modelDest):
	"""
		This function uses Word2Vec to train the model. 
	"""
	command = "./tools/word2vec/word2vec-read-only/word2vec -train <filenameOrig> -output <modelDest> -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1"
	command = command.replace("<filenameOrig>", filenameOrig)
	command = command.replace("<modelDest>", modelDest)
	process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
	output = process.communicate()[0]


def Word2VecGetVector(my_model, word):
	"""
		This function uses Word2Vec to get a vector for a particular word in a particular model . 
	"""
	# I think you can just do my_model[word]
	ret = None
	try:
		ret = my_model.syn0[my_model.vocab[word].index]
	except:
		#sys.stdout.write("couldn't get the word: " + word + "\n")
		return None
	return ret


def Word2VecGetModel(modelPath):
	"""
		This function returns the model stored in the c-format at the given modelPath
		Just a Wrapper Function
	"""
	return word2vec.Word2Vec.load_word2vec_format(modelPath, binary=True)

def GetVectorsForWords(model, list_of_words):
	result = np.zeros([len(list_of_words), model.layer1_size])
	count = 0;
	sys.stdout.write("words processed: " + str(0).zfill(5))
	for w in list_of_words:
		try:
			result[count, :] = model[w.lemma]
		except:
			None
		count = count + 1
		sys.stdout.write("\b\b\b\b\b" + str(count).zfill(5))

	return result

def Word2VecLoadWordsHashTable(model, list_of_words, representation=FeatureRepresentation.STRING):

	list_of_words = set(list_of_words)
	#list_of_words = RemoveItemsWithPOSExcept(list_of_words)
	#number_of_words = len(list_of_words); 
	fpr = None
	counter=0; 

	dictWord2Vec = {}
	for w in list_of_words:
		vec = Word2VecGetVector(model, ConvertUnit(w, FeatureRepresentation.STRING))
		#if word not found, ignore
		if vec is None:
			continue 
		dictWord2Vec[ConvertUnit(w, representation)]=counter
		counter = counter + 1

	number_of_words = counter; 
	counter = 0
	for w in list_of_words:
		vec = Word2VecGetVector(model, ConvertUnit(w, FeatureRepresentation.STRING))
		#if word not found, ignore
		if vec is None:
			continue 
		if fpr is None:
			dimensions = len(vec)
			fpr = np.memmap('file.temp', dtype='float32', mode='w+', shape=(number_of_words,dimensions)); 
			#sys.stdout.write("dimensions: " + str(dimensions))
		fpr[counter] = vec; 
		counter = counter + 1
		
	return (dictWord2Vec, fpr)
