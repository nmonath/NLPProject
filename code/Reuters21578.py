from nltk.corpus import reuters
import os

def SRL(dirname, traintest):
	"""
    	Calls clearnlp parser on all the files in the given directories
  	"""
    os.system("java -XX:+UseConcMarkSweepGC -Xmx3g com.clearnlp.nlp.engine.NLPDecode -z srl -c config_en_srl.xml -i "  + os.path.join(dirname, traintest) + " -oe srl")


def MakeDataSetFiles(dirname):
	 """
	    Creates the data files. Downloads them from the web

	 """
	if not os.path.exists(dirname):
		os.mkdir(dirname)
	if not os.path.exists(os.path.join(dirname, 'train')):
		os.mkdir(os.path.join(dirname, 'train'))
	if not os.path.exists(os.path.join(dirname, 'test')):
		os.mkdir(os.path.join(dirname, 'test'))

	train_classes = file(os.path.join(dirname, 'train_classes.txt'), 'w');
	test_classes = file(os.path.join(dirname, 'test_classes.txt'), 'w');
  

	for fid in reuters.fileids():
		if fid.startswith('training/'):
			filename = 'train_' + fid[9:].zfill(5);
			f = file(os.path.join(dirname, 'train', filename), 'w');
			doc = reuters.open(fid).read()
			f.write(doc);
			f.close()
			for label in reuters.categories(fid):
				train_classes.write(str(reuters.categories().index(label)) + ' ')
			train_classes.write('\n')
		elif fid.startswith('test/'):
			filename = 'test_' + fid[5:].zfill(5);
			f = file(os.path.join(dirname, 'test', filename), 'w');
			doc = reuters.open(fid).read()
			f.write(doc);
			f.close()
			for label in reuters.categories(fid):
				test_classes.write(str(reuters.categories().index(label)) + ' ')
			test_classes.write('\n')


	class_index = file(os.path.join(dirname, 'class_label_index.txt'), 'w')
	for label in reuters.categories():
		class_index.write(label + '\n')
	class_index.close()
	train_classes.close()
	test_classes.close()