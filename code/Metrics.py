import numpy as np
from scipy import stats

def Purity(cluster_labels, true_labels, return_per_cluster=False):
	unique_cluster_labels = np.unique(cluster_labels)
	per_cluster_purity = np.zeros_like(unique_cluster_labels)
	overall_purity = 0
	for i in range(0, unique_cluster_labels.shape[0]):
		lab = unique_cluster_labels[i]
		per_cluster_purity[lab] = ( 1 / np.sum(cluster_labels==lab)) * stats.mode(true_labels[cluster_labels==lab])[1][0]
		overall_purity += stats.mode(true_labels[cluster_labels==lab])[1][0]
	overall_purity = overall_purity / cluster_labels.shape[0]
	if return_per_cluster:
		return (overall_purity, per_cluster_purity)
	else:
		return overall_purity

def NormalizedMutualInformation(cluster_labels, true_labels):
	unique_cluster_labels = np.unique(cluster_labels)
	unique_class_labels = np.unique(true_labels)
	N = np.float32(cluster_labels.shape[0])
	nmi = 0;
	for i in range(0, unique_cluster_labels.shape[0]):
		lab = unique_cluster_labels[i]
		num_in_clust = np.sum(cluster_labels==lab)
		for j in range(0, unique_class_labels.shape[0]):
			clss = unique_class_labels[j]
			num_with_class_label_in_clust = np.sum(true_labels[cluster_labels==lab]==clss)
			print num_with_class_label_in_clust/N
			num_with_class_label = np.sum(true_labels==clss)
			print num_with_class_label/N
			nmi +=  (num_with_class_label_in_clust / N) * np.log((1 + N*num_with_class_label_in_clust) / (num_in_clust * num_with_class_label))

	nmi = nmi / ( (Entropy(cluster_labels) + Entropy(true_labels))/2 )
	return nmi

def Entropy(labels):
	H = 0
	unique_labels = np.unique(labels)
	N = np.float32(labels.shape[0])
	for i in range(0, unique_labels.shape[0]):
		lab = unique_labels[i]
		num_with_lab = np.sum(labels==lab)
		H -= ( num_with_lab/N ) * ( np.log( (1 + num_with_lab)/N ) )
	return H

def RandIndex(cluster_labels, true_labels):
	"""
		Modified from Matlab file exchange: http://www.mathworks.com/matlabcentral/fileexchange/13916-simple-tool-for-estimating-the-number-of-clusters/content/valid_RandIndex.m
	"""

	# Form contingency matrix
	C = np.zeros((np.max(cluster_labels)+1, np.max(true_labels)+1))
	for i in range(0, cluster_labels.shape[0]):
		C[cluster_labels[i], true_labels[i]] = C[cluster_labels[i], true_labels[i]] + 1

	n=np.sum(C);
	nis = np.sum(np.power(np.sum(C, axis=1), 2))
	njs = np.sum(np.power(np.sum(C, axis=0), 2))

	t1=choose(n,2);		
	t2=np.sum(np.power(C, 2))
	t3=0.5*(nis+njs);

	num_agreements = t1+t2-t3
	return num_agreements/t1




def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    From: http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

