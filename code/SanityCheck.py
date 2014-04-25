# Running a Test
import Features


print("The following Test Should Validate that the feature detection is working correctly. This should be run from the code directory")

print("\nLet's start by looking unlemmatized case sensitive words, removing all words without the specified POS tags and words containing symbols or consisting of a single character.\nThe number of times each word appears will be shown in the X matrix")

print("\nHere we do not remove features that appear in only one document and only appearing one time")

Features.USE_LEMMA = False
Features.CASE_SENSITIVE = True
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False

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

print("\nNow, let's try removing features that appear in only one document")

Features.USE_LEMMA = False
Features.CASE_SENSITIVE = True
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = True
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = False


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


print("\nNow, let's try remove features that appear in one time document")

Features.USE_LEMMA = False
Features.CASE_SENSITIVE = True
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = True


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