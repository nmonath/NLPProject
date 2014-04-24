NLPProject
==========

# Dependencies

Numpy and Scipy are required for using our code. We recommend downloading this distribution of Python, which includes all of Numpy and Scipy and many other useful packages: https://store.continuum.io/cshop/anaconda/

You will also need to install the Enum package for Python, which can be done with the command:
```
$ easy_install enum
```
As well as the Gensim package, which can be done with the command:
```
$ easy_install -U gensim
```

Gensim provides an interface with Word2Vec in Python. 

Finally, to perform the Fuzzy Clustering experiments. You'll need to download and install _peach_. You can download it from here: https://code.google.com/p/peach/ Just unzip the directory and place in the folder containing your Python libraries.

# Using Features.py

Import the Features module by

```
import Features
```

The _data\_set_ folders are of the following format:

```
data_set
       <data_set_name>
              train_classes.txt
              test_classes.txt
              class_label_index.txt
              train/
                     train_00001
                     train_00001.srl
              test/
                     test_00001
                     test_00001.srl
```

Each file train/test\_XXXXX is a raw text file containing a training or testing document, train/test\_XXXX.srl, is the dependency parsed and semantic role labeled version of document.  _train\_classes.txt_ and _test\_classes.txt_ store the class labels of the training and testing documents. They are organized such that the class label of the ith file (determined by number after train/test in filename) in the train/test folders is on the ith line of the file. 

To determine the defintion of a feature vector for a set of documents, and to extract feature vectors from all documents, use the following command.

```
data_set_path = '<PATH_TO>/data_set/<DATA_SET_NAME>/train'
(feature_definition, x_train) = Features.Features(data_set_path)
```

Several different feature types can be used. These are determined by the optional arguments passed in to the Features method. There are three sets of parameters to specify

```
funit (FeatureUnit)
frep (FeatureRepresentation)
ftype (FeatureType)
```

FeatureUnit determines the base elements of the vector space model. Right now there are three possible values:

```
FeatureUnit.WORD
FeatureUnit.DEPENDENCY_PAIR
FeatureUnit.BOTH
FeatureUnit.PRED_ARG
FeatureUnit.WORDS_PA
```
_WORD_ provides a traditional bag of words approach. _DEPENDENDENCY\_PAIR_ uses only dependency pairs. _BOTH_ uses both words and dependency pairs. _PRED\_ARG_ uses only predicate argument structures. _WORDS_PA_ uses both words and predicate argument structures. 

There are two options for FeatureRepresentation, which determines whether Hash values are used to reprsent the FeatureUnits or if Strings are used.

```
FeatureRepresentation.HASH
FeatureRepresentation.STRING
```

Using a hashed representation of the characters is much faster, but can lead to some collisions. Reports about the number of collisions on average are coming soon.


There are three options for FeatureType, which determines the values inside of the vector space features.

```
FeatureType.BINARY
FeatureType.TFIDF
FeatureType.COUNT
```

You can also use the following options to change the features:

```
Features.USE_LEMMA = True/False

Features.USE_DEP_TAGS = True/False

Features.USE_POS_TAGS = True/False

Features.USE_ARG_LABELS = True/False

```

These determine if the lemmatized form of words are used and whether meta-data from the parser is used.


If you wanted to use dependency pairs as the feature unit, hashed representations, and term frequency-inverse document frequency, we would do:


```
train_data_set_path = '<PATH_TO>/data_set/<DATA_SET_NAME>/train'
(feature_definition, x_train) = Features.Features(train_data_set_path, funit=Feature.FeatureUnit.DEPENDENCY_PAIR, frep=Feature.FeatureRepresentation.HASH, ftype=Feature.FeatureType.TFIDF)
```

We can pass in a feature definition with the optional argument <feature> to extract feature vectors from the testing documents using the feature defined by the training documents. 

```
test_data_set_path = '<PATH_TO>/data_set/<DATA_SET_NAME>/test'
x_test = Features.Features(test_data_set_path, funit=Feature.FeatureUnit.DEPENDENCY_PAIR, frep=Feature.FeatureRepresentation.HASH, ftype=Feature.FeatureType.TFIDF, feature=feature_definition)
```


