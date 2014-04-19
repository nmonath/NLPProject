import os
import numpy as np
import shutil
from sklearn import preprocessing

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

def LoadClassFile(filename):
	Y = list()
	f = open(filename, 'r')
	for line in f:
		spl = line.split(' ')
		Y.append([int(s) for s in spl])
	lb = preprocessing.LabelBinarizer()
	Yprime = lb.fit_transform(Y)
	if not np.all(np.sum(Yprime, axis=1) == 1):
		return Yprime
	else:
		return np.array([item for sublist in Y for item in sublist], dtype=np.uint8)

def WriteClassFile(class_labels, filename):
	f = file(filename, 'w')
	for c in xrange(0, class_labels.shape[0]):
		if class_labels.ndim > 1:
			f.write(str(class_labels[c,0]))
			for r in xrange(1, class_labels.shape[1]):
				f.write(" " + str(class_labels[c,r]))
		else:
			f.write(str(class_labels[c]))
		f.write("\n")
	f.close()

def SVMLightWrite(targets, features, filename):
	[N,D]=features.shape;
	fid=file(filename,'w');
	for n in xrange(0, N):
		fid.write(str(targets[n]));
		for d in xrange(0,D):
			if (abs(features[n,d])>1e-3):
				fid.write(' ' + str(d+1) + ':' + str(features[n,d])) 
	fid.write('\n');
	fid.close()

def SRL(dirname, traintest):
	"""
    	Calls clearnlp parser on all the files in the given directories
  	"""
  	os.system("java -XX:+UseConcMarkSweepGC -Xmx3g com.clearnlp.nlp.engine.NLPDecode -z srl -c config_en_srl.xml -i "  + os.path.join(dirname, traintest) + " -oe srl")

def FeatureOccuranceReport(feature_def, x):
	"""
		Inputs
			feature_def - the feature definition created using Feature.DefineFeature or the combo method features
			x - an M by N matrix of features using the COUNT option of Features.Features()
		Outputs
			returns an N by 2 numpy array where the first column is the feature definition and the second is the number of occurances of the feature
	"""
	counts = np.sum(x, axis=0)
	count_order = np.argsort(counts)
	feature_def = feature_def[count_order]
	counts = counts[count_order].reshape((count_order.shape[0], 1))
	fd = np.hstack([feature_def, counts])
	return fd


def MakeSmallTest(orig_dir, new_dir,TopK=3,NumDocOfEach=np.inf):
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)
	if not os.path.exists(os.path.join(new_dir, 'train')):
		os.mkdir(os.path.join(new_dir, 'train'))
	if not os.path.exists(os.path.join(new_dir, 'test')):
		os.mkdir(os.path.join(new_dir, 'test'))

	train_classes = file(os.path.join(new_dir, 'train_classes.txt'), 'w');
	test_classes = file(os.path.join(new_dir, 'test_classes.txt'), 'w');
 
	orig_train_classes = LoadClassFile(os.path.join(orig_dir, 'train_classes.txt'))
	print(orig_train_classes.shape)
	orig_test_classes = LoadClassFile(os.path.join(orig_dir, 'test_classes.txt'))

	uniq_class_labels = np.unique(orig_train_classes)
	uniq_class_labels= uniq_class_labels.reshape([uniq_class_labels.shape[0], 1]);
	num_occurances = np.sum(uniq_class_labels == orig_train_classes, axis=1)
	train_count_of_each_in_small_test = np.zeros_like(uniq_class_labels,dtype=np.uint64)
	test_count_of_each_in_small_test = np.zeros_like(uniq_class_labels,dtype=np.uint64)
	cut_off = np.flipud(np.sort(num_occurances))[TopK-1]
	counter = 0
	for filename in os.listdir(os.path.join(orig_dir, 'train')):
		if filename[-4:] == '.srl':
			if num_occurances[orig_train_classes[counter]] >= cut_off and train_count_of_each_in_small_test[orig_train_classes[counter]] < NumDocOfEach:
				shutil.copyfile(os.path.join(orig_dir, 'train', filename), os.path.join(new_dir, 'train', filename))
				train_classes.write(str(orig_train_classes[counter]) + "\n")
				train_count_of_each_in_small_test[orig_train_classes[counter]] = train_count_of_each_in_small_test[orig_train_classes[counter]] + 1
			counter = counter + 1
	counter = 0
	for filename in os.listdir(os.path.join(orig_dir, 'test')):
		if filename[-4:] == '.srl':
			if num_occurances[orig_test_classes[counter]] >= cut_off and test_count_of_each_in_small_test[orig_test_classes[counter]] < NumDocOfEach:
				shutil.copyfile(os.path.join(orig_dir, 'test', filename), os.path.join(new_dir, 'test', filename))
				test_classes.write(str(orig_test_classes[counter]) + "\n")
				test_count_of_each_in_small_test[orig_test_classes[counter]] = test_count_of_each_in_small_test[orig_test_classes[counter]] + 1
			counter = counter + 1
	return (train_count_of_each_in_small_test, test_count_of_each_in_small_test)
