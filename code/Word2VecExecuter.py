#working on it 
import subprocess
from gensim.models import word2vec

#check if use list, or one file, for now only using one file as in demo 
def Word2VecTrain(filenameOrig, modelDest):
	"""
		This function uses Word2Vec to train the model. 
	"""
	command = "./word2vec -train <filenameOrig> -output <modelDest> -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1"
	command = command.replace("<filenameOrig>", filenameOrig)
	command = command.replace("<modelDest>", modelDest)
	process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
	output = process.communicate()[0]


def Word2VecGetVector(modelPath, word):
	"""
		This function uses Word2Vec to get a vector for a particular word in a particular model . 
	"""
	model = word2vec.Word2Vec.load_word2vec_format(modelPath, binary=True)
 	return my_model.syn0[my_model.vocab[word].index]

