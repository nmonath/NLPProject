import numpy as np
from sklearn import cluster, metrics

def Word2VecClustering(idx, embeddings, filename=None):
	dbscan = cluster.DBSCAN(eps=0.5, min_samples=1, metric=metrics.pairwise.cosine_similarity)
	labels = dbscan.fit_predict(embeddings)

	if filename:
		f = open(filename, 'w')
		words = np.array(idx.keys(), dtype=np.object)
		for l in np.unique(labels):
			w_in_cluster = words[labels==l]
			f.write('#' + str(l)+ ': {' + str(w_in_cluster[0]))
			for w in range(1, len(w_in_cluster)):
				f.write(", " + w_in_cluster[w])
			f.write("}\n")
		f.close()

	return labels