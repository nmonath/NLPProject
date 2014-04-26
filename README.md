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

Finally, to perform the Fuzzy Clustering experiments. You'll need to download and install **peach**. You can download it from here: https://code.google.com/p/peach/ Just unzip the directory and place in the folder containing your Python libraries.

# The Features Module

The **Features** module is used to extract Vector Space Model feature vectors from parsed text documents. The parsing comes as a preprocessing step using **ClearNLP**'s parser (this will be explained below). The **Features** module allows for several different _flavors_ of features in addition to the traditional bag-of-words features. It also allows for several different options such as the use of lemmatization, the inclusion of part-of-speech tags, etc. 

Let's begin by defining some terminology that will make this explanation more clear. By _units_ or _base units_ we are referring to the terms whose presence/absense determine in a document determine the values of the feature vector of the document. This means each _unit_ corresponds to an entry in the feature vector of a document. The different possible _units_ will be described below. We'll call the _feature definition_ the list of _units_ corresponding to the entries in a _feature vector_ of a document. The typical definition of _feature vector_ is intented here. A _feature vector_ is a D-element vector such that element _i_ corresponds to the _ith_ element of the _feature definition_ with a value representing the presence/absense of the _unit_. We say that _units_ and their corresponding _feature definition_ can have one of two _representations_. The _representation_ determines if the hashed version of the _unit_ or the string version of the _unit_. Finally, a _feature vector_ can be of one of three _types_. It can be _binary_, which means the values of the _feature vector_ are 0 or 1 depending on whether or not a _unit_ appears in the document. It can be _tf-idf_, in which the values of the _feature vector_ are the term-frequency-inverse-document-frequency of a _unit_. Lastly, it can be _count_, in which the values of _feature vector_ are the term frequencies of the _units_. 

Now let's examine the different _units_ that the **Features** module provides. There are three base forms of _units_, _words_, _dependency pairs_, and _predicate argument components_. These forms can also be combined together, e.g. _words and depedency pairs_, _dependency pairs and predicate argument components_, _all three_, etc. By _words_, we mean a traditional bag of words representation. _Dependency pairs_ refers to the bag of all depedency pairs of words (as determined by the dependency parser). _Predicate argument components_ refers to the bag of all the predicates and arguments that appear in a document. **Note** that this does not mean that a predicate argument structure is the _unit_. It means that the predicate itself is one _unit_ and each of the arguments are additional _units_. For example, if we had the predicate argument structure: {Predicate: ate, Arg0: the hungry child, Arg1: the cake}. The _feature definition_ would have three _units_: "ate", "the hungry child" and "the cake". 

To select which _unit_, _representation_, and _type_ are used to extract features from documents. The following global variables of the module are used:

```
FUNIT
FREP
FTYPE
```

The values that these variables take on are ``` enum ``` values from the classes ```FeatureUnits```, ```FeatureRepresentation``` , ```FeatureType```, which are nested in the features module. Below are the possible values for each variable:

```Python
import Features

Features.FUNIT = Features.FeatureUnits.WORD
Features.FUNIT = Features.FeatureUnits.DEPENDENCY_PAIR
Features.FUNIT = Features.FeatureUnits.WORDS_AND_DEPENDENCY_PAIRS
Features.FUNIT = Features.FeatureUnits.PREDICATE_ARGUMENT 
Features.FUNIT = Features.FeatureUnits.WORDS_AND_PREDICATE_ARGUMENT
Features.FUNIT = Features.FeatureUnits.DEPENDENCY_PAIRS_AND_PREDICATE_ARGUMENT 
Features.FUNIT = Features.FeatureUnits.ALL (default)

Features.FREP = Features.FeatureRepresentation.HASH (default)
Features.FREP = Features.FeatureRepresentation.STRING

Features.FTYPE = Features.FeatureType.BINARY 
Features.FTYPE = Features.FeatureType.TFIDF (default)
Features.FTYPE = Features.FeatureType.COUNT 
```

Now let's go over some of the other options we can set in the **Features** module. We can determine if the _units_ are lemmatized with the following global variable:

```Python
Features.USE_LEMMA = True (default) /False
```

We can determine if _units_ are case sensitive. Of course if lemmatization is used the value of this variable will have no effect.

```Python
Features.CASE_SENSITIVE = True/False (default)
```

We can append part of speech tags (provided by the parser) to every word in every _unit_ with the following global variable:

```Python
Features.USE_POS_TAGS = True/False (default)
```

The dependency parser also provides _dependency relation tags_. These can be appended to words in ```FeatureUnits.DEPENDENCY_PAIR``` units with the following global variable:

```Python
Features.USE_DEP_TAGS = True/False (default)
```

The semantic role labeler also provides argument labels. These can be appended to the argument structures. This can be set with the following global variable:


```Python
Features.USE_ARG_LABELS  = True/False (default)
```

**NOTE** Appending any of these labels will result in a larger feature space. It will create a unique _unit_ for each word and argument labels.


Rather than removing stop words based on a fixed list, we only keep those words with certain part of speech tags. The part of speech tags which define the retained word are controlled by the global ```KEEPER_POS```. It's default value is shown below:

```Python
KEEPER_POS = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RR", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
```

**NOTE**. A dependency pair _unit_ is retained only if both words in the pair have part of speech tags in ```KEEPER_POS```. Predicates are retained if its POS tag appears in ```KEEPER_POS```. Only the words in an argument that have POS tags in ```KEEPER_POS``` are retained. This means that we had the following predicate argument structure: {Predicate: drove, Arg0: the old man, Arg1: there}. The only units that would be retained are "drove" and "old man". "the" is dropped from "the old man" and "there" is left off entirely.

We also provide four other options to remove _units_ which might introduce noise. These options are to remove units consisting of a single character, to remove _units_ which appear only one time, to remove _units_ that appear in only one document, and to remove _units_ containing non-alpha-numeric symbols. The first three are set with the following global variables.

```
Features.REMOVE_SINGLE_CHARACTERS = True (default) / False
Features.REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT = True (default) / False
Features.REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME = True (default) / False
```

The final option, the removal of _units_ with non-alpha numeric symbols, is generalized to only keeping _units_ consisting of a specified set of symbols. The global variable ```SYMBOLS_TO_KEEP``` is a regular expression (an ```re``` in Python) which specifies the set of symbols from which a _unit_ must be drawn for the _unit_ to be a part of the feature definition. It's default value is shown below. 

```
Features.SYMBOLS_TO_KEEP = '[a-zA-Z0-9]*'
```

There is also one optimization configuration option, the use of memory maps for the feature vector matrices. This can be set with the following global:

```
Features.USE_MEMORY_MAP = True / False (default)
```

The current status of each variable can be quickly checked using the ```DisplayConfigurations()``` method of the **Features** module. 

```
Features.DisplayConfiguration()
```

which gives the following output:

```
Feature Configuration Settings
------------------------------
USE_LEMMA: True
CASE_SENSITIVE: False
USE_POS_TAGS: False
USE_DEP_TAGS: False
USE_ARG_LABELS: False
SYMBOLS_TO_KEEP: [a-zA-Z0-9]*
REMOVE_SINGLE_CHARACTERS: True
REMOVE_FEATURES_APPEARING_IN_ONLY_ONE_DOCUMENT: True
REMOVE_FEATURES_ONLY_APPEARING_ONE_TIME: True
USE_MEMORY_MAP: False
KEEPER_POS: ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RR', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
FUNIT: Words, Dependency Pairs, and Predicate Argument Components
FREP: Hash
FTYPE: TF-IDF
```

# Using SupervisedLearning.py and UnsupervisedLearning.py

These modules allow you to perform classification and clustering experiments on data sets formatted in the following way:

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

In order to use these modules, you first need to import the Features module:

```
import Features
```

Now, configure the Features module the way you would like to extract features. There are several different options. The first decision you should make is what the base "unit" of the your vector space model is. The base unit can be words, dependency pairs, or the predicates and arguments discovered via semantic role labeling, or any combination of these. To select which base unit is used, set the FUNIT global variable of the features model to one of the values in the Features.FeatureUnits enum. For example,

```
Features.FUNIT = Features.FeatureUnits.WORD
Features.FUNIT = Features.FeatureUnits.DEPENDENCY_PAIR
Features.FUNIT = Features.FeatureUnits.WORDS_AND_DEPENDENCY_PAIRS
Features.FUNIT = Features.FeatureUnits.PREDICATE_ARGUMENT 
Features.FUNIT = Features.FeatureUnits.WORDS_AND_PREDICATE_ARGUMENT
Features.FUNIT = Features.FeatureUnits.DEPENDENCY_PAIRS_AND_PREDICATE_ARGUMENT 
Features.FUNIT = Features.FeatureUnits.ALL 
```


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


