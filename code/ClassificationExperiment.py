
def run(dirname):
	# Run a test on news groups

	# Always Use Hashing
	# Do 
	# Binary - Word
	# Binary - DP
	# Binary - Both
	# TFIDF - Word
	# TFIDF - DP
	# TFIDF - Both

	import Evaluation
	import Features
	import os
	import Util
	import numpy as np
	from scipy.sparse import csr_matrix

	print('*' * 80)
	print(' ' * 37 + 'BINARY' + ' ' * 37)
	print(' ' * 38 + 'WORD' + ' ' * 38)
	print('*' * 80)

	y_train = Util.LoadClassFile(os.path.join(dirname, 'train_classes.txt'))
	y_test = Util.LoadClassFile(os.path.join(dirname, 'test_classes.txt'))
	categories = Util.LoadClassLabels(os.path.join(dirname, 'class_label_index.txt'))
	 
	if not os.path.exists('training_and_testing_files'):
		os.mkdir('training_and_testing_files')
	(feature_def_word, x_train_word) = Features.Features(os.path.join(dirname, 'train'), ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.WORD,frep=Features.FeatureRepresentation.HASH)
	x_test_word = Features.Features(os.path.join(dirname, 'test'), ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.WORD,frep=Features.FeatureRepresentation.HASH, feature=feature_def_word)
	print(' ')

	np.save(os.path.join('training_and_testing_files', 'feature_def_word.npy'), feature_def_word)
	np.save(os.path.join('training_and_testing_files', 'x_train_word.npy'), x_train_word)

	binary_x_train_word = csr_matrix(Features.ToBINARY(x_train_word))
	binary_x_test_word = csr_matrix(Features.ToBINARY(x_test_word))

	Evaluation.Evaluate(binary_x_train_word, y_train, binary_x_test_word, y_test, categories=categories)

	binary_x_train_word = 0;
	binary_x_test_word = 0;
	feature_def_word = 0;

	print('*' * 80)
	print(' ' * 37 + 'TFIDF' + ' ' * 38)
	print(' ' * 38 + 'WORD' + ' ' * 38)
	print('*' * 80)

	tfidf_x_train_word = csr_matrix(Features.ToTFIDF(x_train_word))
	tfidf_x_test_word = csr_matrix(Features.ToTFIDF(x_test_word))
	x_train_word = 0;
	x_test_word = 0;

	Evaluation.Evaluate(tfidf_x_train_word, y_train, tfidf_x_test_word, y_test, categories=categories)

	tfidf_x_train_word = 0;
	tfidf_x_test_word = 0;



	print('*' * 80)
	print(' ' * 37 + 'BINARY' + ' ' * 37)
	print(' ' * 38 + 'BOTH' + ' ' * 38)
	print('*' * 80)

	if not os.path.exists('training_and_testing_files'):
		os.mkdir('training_and_testing_files')
	(feature_def_both, x_train_both) = Features.Features(os.path.join(dirname, 'train'), ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.BOTH,frep=Features.FeatureRepresentation.HASH)
	x_test_both = Features.Features(os.path.join(dirname, 'test'), ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.BOTH,frep=Features.FeatureRepresentation.HASH, feature=feature_def_both)
	print(' ')

	np.save(os.path.join('training_and_testing_files', 'feature_def_both.npy'), feature_def_both)
	np.save(os.path.join('training_and_testing_files', 'x_train_both.npy'), x_test_both)

	binary_x_train_both = csr_matrix(Features.ToBINARY(x_train_both))
	binary_x_test_both = csr_matrix(Features.ToBINARY(x_test_both))

	Evaluation.Evaluate(binary_x_train_both, y_train, binary_x_test_both, y_test, categories=categories)


	binary_x_train_both = 0;
	binary_x_test_both = 0;
	feature_def_both = 0;

	print('*' * 80)
	print(' ' * 37 + 'TFIDF' + ' ' * 38)
	print(' ' * 38 + 'BOTH' + ' ' * 38)
	print('*' * 80)



	tfidf_x_train_both = csr_matrix(Features.ToTFIDF(x_train_both))
	tfidf_x_test_both = csr_matrix(Features.ToTFIDF(x_test_both))
	x_train_both = 0;
	x_test_both = 0;

	Evaluation.Evaluate(tfidf_x_train_both, y_train, tfidf_x_test_both, y_test, categories=categories)

	tfidf_x_train_both = 0;
	tfidf_x_test_both = 0;



	print('*' * 80)
	print(' ' * 37 + 'BINARY' + ' ' * 37)
	print(' ' * 38 + 'DEPENDENCY_PAIR' + ' ' * 38)
	print('*' * 80)

	if not os.path.exists('training_and_testing_files'):
		os.mkdir('training_and_testing_files')
	(feature_def_DEPENDENCY_PAIR, x_train_DEPENDENCY_PAIR) = Features.Features(os.path.join(dirname, 'train'), ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.DEPENDENCY_PAIR,frep=Features.FeatureRepresentation.HASH)
	x_test_DEPENDENCY_PAIR = Features.Features(os.path.join(dirname, 'test'), ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.DEPENDENCY_PAIR,frep=Features.FeatureRepresentation.HASH, feature=feature_def_DEPENDENCY_PAIR)
	print(' ')

	np.save(os.path.join('training_and_testing_files', 'feature_def_DEPENDENCY_PAIR.npy'), feature_def_DEPENDENCY_PAIR)
	np.save(os.path.join('training_and_testing_files', 'x_train_DEPENDENCY_PAIR.npy'), x_test_DEPENDENCY_PAIR)

	binary_x_train_DEPENDENCY_PAIR = csr_matrix(Features.ToBINARY(x_train_DEPENDENCY_PAIR))
	binary_x_test_DEPENDENCY_PAIR = csr_matrix(Features.ToBINARY(x_test_DEPENDENCY_PAIR))

	Evaluation.Evaluate(binary_x_train_DEPENDENCY_PAIR, y_train, binary_x_test_DEPENDENCY_PAIR, y_test, categories=categories)

	binary_x_train_DEPENDENCY_PAIR = 0;
	binary_x_test_DEPENDENCY_PAIR = 0;
	feature_def_DEPENDENCY_PAIR = 0;

	print('*' * 80)
	print(' ' * 37 + 'TFIDF' + ' ' * 38)
	print(' ' * 38 + 'DEPENDENCY_PAIR' + ' ' * 38)
	print('*' * 80)

	tfidf_x_train_DEPENDENCY_PAIR = csr_matrix(Features.ToTFIDF(x_train_DEPENDENCY_PAIR))
	tfidf_x_test_DEPENDENCY_PAIR = csr_matrix(Features.ToTFIDF(x_test_DEPENDENCY_PAIR))
	x_train_DEPENDENCY_PAIR = 0;
	x_test_DEPENDENCY_PAIR = 0;
	Evaluation.Evaluate(tfidf_x_train_DEPENDENCY_PAIR, y_train, tfidf_x_test_DEPENDENCY_PAIR, y_test, categories=categories)

	tfidf_x_train_DEPENDENCY_PAIR = 0;
	tfidf_x_test_DEPENDENCY_PAIR = 0;






