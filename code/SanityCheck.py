# Running a Test
import Features

print('%' * 80)
print('%' * 80 + "\n")
print("WORDS\n")
print('%' * 80 )
print('%' * 80 + "\n")

print('%' * 80 + "\n")

print("The following Test Should Validate that the feature detection is working correctly. This should be run from the code directory")

print("\nLet's start by looking unlemmatized case sensitive words, removing all words without the specified POS tags and words containing symbols or consisting of a single character.\nThe number of times each word appears will be shown in the X matrix")

print("\nHere we do not remove features that appear in only one document and only appearing one time\n")

print('%' * 80 + "\n")

Features.USE_LEMMA = False
Features.CASE_SENSITIVE = True
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.WORD,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: " + file('../data_sets/unit_test2/train/train_00001').read()
doc2c = "Document 2: " + file('../data_sets/unit_test2/train/train_00002').read()
doc3c = "Document 3: " + file('../data_sets/unit_test2/train/train_00003').read()
doc4c = "Document 4: " +file('../data_sets/unit_test2/train/train_00004').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")

print('%' * 80 + "\n")
print("\nNow, let's try removing features that appear in only one document\n")
print('%' * 80 + "\n")

Features.USE_LEMMA = False
Features.CASE_SENSITIVE = True
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = True
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
Features.DisplayConfiguration()



(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.WORD,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: " + file('../data_sets/unit_test2/train/train_00001').read()
doc2c = "Document 2: " + file('../data_sets/unit_test2/train/train_00002').read()
doc3c = "Document 3: " + file('../data_sets/unit_test2/train/train_00003').read()
doc4c = "Document 4: " +file('../data_sets/unit_test2/train/train_00004').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")

print('%' * 80 + "\n")
print("\nNow, let's try remove features that appear only one time \n")
print('%' * 80 + "\n")

Features.USE_LEMMA = False
Features.CASE_SENSITIVE = True
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = True
Features.DisplayConfiguration()



(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.WORD,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: " + file('../data_sets/unit_test2/train/train_00001').read()
doc2c = "Document 2: " + file('../data_sets/unit_test2/train/train_00002').read()
doc3c = "Document 3: " + file('../data_sets/unit_test2/train/train_00003').read()
doc4c = "Document 4: " +file('../data_sets/unit_test2/train/train_00004').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")

print('%' * 80 + "\n")
print("\nNow, let's make the features case insensitive. Including features only appearing one time and in one document\n")
print('%' * 80 + "\n")


Features.USE_LEMMA = False
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
Features.DisplayConfiguration()



(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.WORD,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: " + file('../data_sets/unit_test2/train/train_00001').read()
doc2c = "Document 2: " + file('../data_sets/unit_test2/train/train_00002').read()
doc3c = "Document 3: " + file('../data_sets/unit_test2/train/train_00003').read()
doc4c = "Document 4: " +file('../data_sets/unit_test2/train/train_00004').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")

print('%' * 80 + "\n")
print("\nFinally let's lemmatize the words. Including features only appearing one time and in one document\n")
print('%' * 80 + "\n")


Features.USE_LEMMA = True
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.WORD,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: " + file('../data_sets/unit_test2/train/train_00001').read()
doc2c = "Document 2: " + file('../data_sets/unit_test2/train/train_00002').read()
doc3c = "Document 3: " + file('../data_sets/unit_test2/train/train_00003').read()
doc4c = "Document 4: " +file('../data_sets/unit_test2/train/train_00004').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")

print('%' * 80 + "\n")
print("\nOne more thing with words. Let's check that lemmatized, removing features that appear 1 time or in 1 document works\n")
print('%' * 80 + "\n")


Features.USE_LEMMA = True
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = True
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = True
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.WORD,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: " + file('../data_sets/unit_test2/train/train_00001').read()
doc2c = "Document 2: " + file('../data_sets/unit_test2/train/train_00002').read()
doc3c = "Document 3: " + file('../data_sets/unit_test2/train/train_00003').read()
doc4c = "Document 4: " +file('../data_sets/unit_test2/train/train_00004').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")


print('%' * 80)
print('%' * 80 + "\n")
print("DEPENDENCY_PAIRS\n")
print('%' * 80 )
print('%' * 80 + "\n")

print("\n")
print('%' * 80 + "\n")
print("First let's make a feature with all the dependency pairs occuring in the documents. The pairs are case sensitive and use lemmatization.")
print('%' * 80 + "\n")



Features.USE_LEMMA = False
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.DEPENDENCY_PAIR,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: \n" + file('../data_sets/unit_test2/train/train_00001.srl').read()
doc2c = "Document 2: \n" + file('../data_sets/unit_test2/train/train_00002.srl').read()
doc3c = "Document 3: \n" + file('../data_sets/unit_test2/train/train_00003.srl').read()
doc4c = "Document 4: \n" +file('../data_sets/unit_test2/train/train_00004.srl').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")


print('%' * 80 + "\n")
print("Now let's remove dependency pairs that appear only one time.")
print('%' * 80 + "\n")



Features.USE_LEMMA = False
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = True
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.DEPENDENCY_PAIR,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: \n" + file('../data_sets/unit_test2/train/train_00001.srl').read()
doc2c = "Document 2: \n" + file('../data_sets/unit_test2/train/train_00002.srl').read()
doc3c = "Document 3: \n" + file('../data_sets/unit_test2/train/train_00003.srl').read()
doc4c = "Document 4: \n" +file('../data_sets/unit_test2/train/train_00004.srl').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")


print('%' * 80 + "\n")
print("Now let's remove dependency pairs that appear only one document")
print('%' * 80 + "\n")



Features.USE_LEMMA = False
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = True
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.DEPENDENCY_PAIR,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: \n" + file('../data_sets/unit_test2/train/train_00001.srl').read()
doc2c = "Document 2: \n" + file('../data_sets/unit_test2/train/train_00002.srl').read()
doc3c = "Document 3: \n" + file('../data_sets/unit_test2/train/train_00003.srl').read()
doc4c = "Document 4: \n" +file('../data_sets/unit_test2/train/train_00004.srl').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")


print('%' * 80 + "\n")
print("Now let's remove dependency pairs that appear only one document and only one time")
print('%' * 80 + "\n")



Features.USE_LEMMA = False
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = True
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = True
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.DEPENDENCY_PAIR,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: \n" + file('../data_sets/unit_test2/train/train_00001.srl').read()
doc2c = "Document 2: \n" + file('../data_sets/unit_test2/train/train_00002.srl').read()
doc3c = "Document 3: \n" + file('../data_sets/unit_test2/train/train_00003.srl').read()
doc4c = "Document 4: \n" +file('../data_sets/unit_test2/train/train_00004.srl').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")


print('%' * 80 + "\n")
print("Finally let's lemmatize the features. Including those that appear in only one document or only one time.\n")
print('%' * 80 + "\n")



Features.USE_LEMMA = True
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.DEPENDENCY_PAIR,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: \n" + file('../data_sets/unit_test2/train/train_00001.srl').read()
doc2c = "Document 2: \n" + file('../data_sets/unit_test2/train/train_00002.srl').read()
doc3c = "Document 3: \n" + file('../data_sets/unit_test2/train/train_00003.srl').read()
doc4c = "Document 4: \n" +file('../data_sets/unit_test2/train/train_00004.srl').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")


print('%' * 80)
print('%' * 80 + "\n")
print("WORDS AND DEPENDENCY_PAIRS\n")
print('%' * 80 )
print('%' * 80 + "\n")

print('%' * 80 + "\n")
print("We just show the lemmatized version. But include features appearing in only one document and only one time")
print('%' * 80 + "\n")

Features.USE_LEMMA = True
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.BOTH,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: \n" + file('../data_sets/unit_test2/train/train_00001.srl').read()
doc2c = "Document 2: \n" + file('../data_sets/unit_test2/train/train_00002.srl').read()
doc3c = "Document 3: \n" + file('../data_sets/unit_test2/train/train_00003.srl').read()
doc4c = "Document 4: \n" +file('../data_sets/unit_test2/train/train_00004.srl').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")



print('%' * 80)
print('%' * 80 + "\n")
print("Predicate Argument Structures.\n")
print('%' * 80 )
print('%' * 80 + "\n")

print("\n")
print('%' * 80 + "\n")
print("First let's make a feature with all the predicate argument occuring in the documents. The PAs are case sensitive and use lemmatization.")
print('%' * 80 + "\n")



Features.USE_LEMMA = False
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.PRED_ARG,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: \n" + file('../data_sets/unit_test2/train/train_00001.srl').read()
doc2c = "Document 2: \n" + file('../data_sets/unit_test2/train/train_00002.srl').read()
doc3c = "Document 3: \n" + file('../data_sets/unit_test2/train/train_00003.srl').read()
doc4c = "Document 4: \n" +file('../data_sets/unit_test2/train/train_00004.srl').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")


print("\n")
print('%' * 80 + "\n")
print("Now we remove those features that appear in only one document")
print('%' * 80 + "\n")



Features.USE_LEMMA = False
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = True
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.PRED_ARG,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: \n" + file('../data_sets/unit_test2/train/train_00001.srl').read()
doc2c = "Document 2: \n" + file('../data_sets/unit_test2/train/train_00002.srl').read()
doc3c = "Document 3: \n" + file('../data_sets/unit_test2/train/train_00003.srl').read()
doc4c = "Document 4: \n" +file('../data_sets/unit_test2/train/train_00004.srl').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")

print('%' * 80 + "\n")
print("Now we remove those features that appear only one time")
print('%' * 80 + "\n")



Features.USE_LEMMA = False
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = True
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.PRED_ARG,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: \n" + file('../data_sets/unit_test2/train/train_00001.srl').read()
doc2c = "Document 2: \n" + file('../data_sets/unit_test2/train/train_00002.srl').read()
doc3c = "Document 3: \n" + file('../data_sets/unit_test2/train/train_00003.srl').read()
doc4c = "Document 4: \n" +file('../data_sets/unit_test2/train/train_00004.srl').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")


print('%' * 80 + "\n")
print("Now we lemmatize the words, but include things that appear only once or in one document")
print('%' * 80 + "\n")



Features.USE_LEMMA = True
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.PRED_ARG,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: \n" + file('../data_sets/unit_test2/train/train_00001.srl').read()
doc2c = "Document 2: \n" + file('../data_sets/unit_test2/train/train_00002.srl').read()
doc3c = "Document 3: \n" + file('../data_sets/unit_test2/train/train_00003.srl').read()
doc4c = "Document 4: \n" +file('../data_sets/unit_test2/train/train_00004.srl').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")

print('%' * 80 + "\n")
print("Now we lemmatize the words, AND REMOVE things that appear only once or in one document")
print('%' * 80 + "\n")



Features.USE_LEMMA = True
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = True
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = True
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.PRED_ARG,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: \n" + file('../data_sets/unit_test2/train/train_00001.srl').read()
doc2c = "Document 2: \n" + file('../data_sets/unit_test2/train/train_00002.srl').read()
doc3c = "Document 3: \n" + file('../data_sets/unit_test2/train/train_00003.srl').read()
doc4c = "Document 4: \n" +file('../data_sets/unit_test2/train/train_00004.srl').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")


print('%' * 80 + "\n")
print("Finaly we include the POS, DepRel, and ArgLabel tags. This should show that it works reguardless of the feature")
print('%' * 80 + "\n")



Features.USE_LEMMA = True
Features.USE_DEP_TAGS = True
Features.USE_POS_TAGS = True
Features.USE_ARG_LABELS = True
Features.CASE_SENSITIVE = False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False
Features.DisplayConfiguration()


(fdef, X) = Features.Features('../data_sets/unit_test2/train/', ftype=Features.FeatureType.COUNT,funit=Features.FeatureUnits.PRED_ARG,frep=Features.FeatureRepresentation.STRING)

print("\nFeature Definition")
print(fdef)
print("\nX matrix")
print(X)

print("\nThis should make sense given the following contents of the documents\n")

doc1c = "Document 1: \n" + file('../data_sets/unit_test2/train/train_00001.srl').read()
doc2c = "Document 2: \n" + file('../data_sets/unit_test2/train/train_00002.srl').read()
doc3c = "Document 3: \n" + file('../data_sets/unit_test2/train/train_00003.srl').read()
doc4c = "Document 4: \n" +file('../data_sets/unit_test2/train/train_00004.srl').read()

print(doc1c)
print(doc2c)
print(doc3c)
print(doc4c)
print("\n")




