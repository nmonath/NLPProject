#working on it 
import subprocess
from gensim.models import word2vec
import numpy as np

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


def Word2VecGetVector(modelPath, word):
	"""
		This function uses Word2Vec to get a vector for a particular word in a particular model . 
	"""
	my_model = word2vec.Word2Vec.load_word2vec_format(modelPath, binary=True)
 	# I think you can just do my_model[word]
 	return my_model.syn0[my_model.vocab[word].index]

def Word2VecGetModel(modelPath):
	"""
		This function returns the model stored in the c-format at the given modelPath
		Just a Wrapper Function
	"""
	return word2vec.Word2Vec.load_word2vec_format(modelPath, binary=True)

def GetVectorsForWords(model, list_of_words):
	result = np.zeros([len(list_of_words), model.layer1_size])
	count = 0;
	for w in list_of_words:
		try:
			result[count, :] = model[w.lemma]
		except:
			None
		count = count + 1
		sys.stdout.write("\b\b\b\b\b" + str(count).zfill(5))
		
	return result
