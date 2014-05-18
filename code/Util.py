import os
import numpy as np
import shutil
from sklearn import preprocessing
import glob
import re 

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

def WriteClassLabelFile(filename, labels):
	f = open(filename, 'w')
	for l in labels:
		f.write(str(l) + '\n')
	f.close()

def LoadClassLabels(filename):
	y_label = list()
	f = open(filename, 'r')
	for line in f:
		y_label.append(line.strip())
	return y_label

def LoadClassFile(filename, out_multi_class_matrix=True):
	Y = list()
	f = open(filename, 'r')
	for line in f:
		spl = line.strip().split(' ')
		Y.append([int(s) for s in spl])
	if not out_multi_class_matrix:
		return Y
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


def preprocessFiles(dirname):
	"""
	This function pre-processes files with the following rules: 
        1- If the file has chains of words larger than 50 characters, it is shown in the screen to see what to do. 
        2- If the file has character \n> or \n>>, the > and >> are deleted. 
	"""
	files = glob.glob(dirname + "/*")
	filesModified = 0
	for file in files:
		cntTimesBigWord = 0
		printed = 0 
		fName = file.split("/")

#		print ("opening: " + dirname + "/modified_" + fName[len(fName)-1])

		pat = re.compile(r'[ABCDEFGHIJKLMNOPQRSTUVWXYZ]')
		patMin = re.compile(r'[abcdefghijklmnopqrstuvwxyz]')
		if(file[len(file)-3:len(file)]!='srl' and os.path.isfile(file)):
#			print file
			writeFile = open(dirname + "/" + fName[len(fName)-1] + "_modified",'w')
			f = open(file)
			prevSusp = 0
			for line in iter(f):
#				charChange=0
#				onlySpaces=1
#				prevI = ""
#				for i in range(0, len(line)):
#					if(prevI != "" and line[i]!= prevI and line[i]!="\n"):
#						charChange=1
#					if(line[i]!=' ' and line[i]!='\n' and ord(line[i])!='\t'):
#						onlySpaces=0
#					if(line[i]!=' '):
#						prevI = line[i]
#				if(charChange==0 and len(line)>5 and onlySpaces==0):
#					print ("\nsame char line: " + line)
                
				splitted = line.split(' ')
				for word in splitted:
					if(len(word)>50):
						#don't write anything to writeFile
						if(cntTimesBigWord==0):
							print ("file modified: " + file)
							filesModified+=1
						cntTimesBigWord+=1
						prevSusp = 1
					elif((len(word)>15 and len(patMin.findall(word))==0 and len(pat.findall(word))/len(word)<0.7) or 
						#(prevSusp==1 and len(patMin.findall(word))==0 and len(pat.findall(word))!=0 and len(pat.findall(word))<len(word))):
						(prevSusp==1 and len(patMin.findall(word))==0 and len(pat.findall(word))<len(word))):
						if(cntTimesBigWord==0):
							print ("file modified: " + file)
							filesModified+=1
						cntTimesBigWord+=1
						prevSusp=1
					else:
						if(word=='\n' or word.find('\n')!=-1):
							writeFile.write(word)
						else:
							writeFile.write(word + " ")
						if(len(word.strip())>0):
							prevSusp=0
#					if(len(word)>50 and cntTimesBigWord>10 and printed == 0):
#						print word
#						print file
#						printed =1
                    
			f.close()
	print("files modified: " + str(filesModified))


def showFilesContent(dirname):
	"""
	This function just displays the files' content inside the directory
	"""
	files = glob.glob(dirname + "/*")
	for file in files:
		cntTimesBigWord = 0
		printed = 0 
		if(file[len(file)-3:len(file)]!='srl'):
			print file
			f = open(file)
			for line in iter(f):
#				line = "------------"
#                if(len(line)>0):
				print line
			f.close()


def SRL(dirname):
	"""
    	Calls clearnlp parser on all the files in the given directories
  	"""
  	os.system("java -XX:+UseConcMarkSweepGC -Xmx3g com.clearnlp.nlp.engine.NLPDecode -z srl -c config_en_srl.xml -i "  + dirname + " -oe srl")
  	for filename in os.listdir(dirname):
  		if filename.startswith('.') and filename[-4:] == '.srl':
  			os.remove(os.path.join(dirname, filename))


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

	class_labels = np.array(LoadClassLabels(os.path.join(orig_dir, 'class_label_index.txt')), dtype=np.object)


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

	new_class_labels = class_labels[num_occurances >= cut_off]
	WriteClassLabelFile(os.path.join(new_dir, 'class_label_index.txt'), new_class_labels)

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







def MakeReutersTest(orig_dir, new_dir, classification=False):
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)
	if not os.path.exists(os.path.join(new_dir, 'train')):
		os.mkdir(os.path.join(new_dir, 'train'))
	if not os.path.exists(os.path.join(new_dir, 'test')):
		os.mkdir(os.path.join(new_dir, 'test'))

	most_freq = ['acq', 'corn', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade', 'wheat']

	class_labels = LoadClassLabels(os.path.join(orig_dir, 'class_label_index.txt'))
	most_freq_idx = [class_labels.index(x) for x in most_freq]

	train_classes = file(os.path.join(new_dir, 'train_classes.txt'), 'w');
	test_classes = file(os.path.join(new_dir, 'test_classes.txt'), 'w');
	orig_train_classes = LoadClassFile(os.path.join(orig_dir, 'train_classes.txt'), out_multi_class_matrix=False)
	orig_test_classes = LoadClassFile(os.path.join(orig_dir, 'test_classes.txt'), out_multi_class_matrix=False)


	counter = 0
	for filename in os.listdir(os.path.join(orig_dir, 'train')):
		if filename[-4:] == '.srl':
			overlap = set(most_freq_idx).intersection(set(orig_train_classes[counter]))
			if overlap and ((not classification) or len(overlap)==1):
				shutil.copyfile(os.path.join(orig_dir, 'train', filename), os.path.join(new_dir, 'train', filename))
				for l in overlap:
					train_classes.write(str(l) + " ")
				train_classes.write("\n")
			counter = counter + 1
	counter = 0
 	for filename in os.listdir(os.path.join(orig_dir, 'test')):
		if filename[-4:] == '.srl':
			overlap = set(most_freq_idx).intersection(set(orig_test_classes[counter]))
			if overlap and ((not classification) or len(overlap)==1):
				shutil.copyfile(os.path.join(orig_dir, 'test', filename), os.path.join(new_dir, 'test', filename))
				for l in overlap:
					test_classes.write(str(l) + " ")
				test_classes.write("\n")
			counter = counter + 1


	WriteClassLabelFile(os.path.join(new_dir, 'class_label_index.txt'), most_freq)

def get_num_samples(dirname):
	count = 0
	for filename in os.listdir(dirname):
		if '.srl' in filename:
			count = count + 1
	return count


#showFilesContent("/Users/klimzaporojets/klim/umass/691CL/NLPProject/data_sets/20_news_groups_classification/train/")
preprocessFiles("/Users/klimzaporojets/klim/umass/691CL/NLPProject/data_sets/20_news_groups_classification/train/")