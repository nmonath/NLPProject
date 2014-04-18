import numpy as np
from sklearn import cluster

def Word2VecClustering(model, words):
	emb = np.zeros((len(words), model.layer1_size))
	count = 0
	for w in words:
		emb[count, :] = model[w]
		count = count + 1
	dbscan = cluster.DBSCAN(eps=1, min_samples=1)
	labels = dbscan.fit_predict(emb)
	return labels