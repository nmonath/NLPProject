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

# Preliminaries
## Data Set Organization

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

## Utility Functions

The **Util** module provides several key functions that will be used throughout this explanation. These functions are used to perform the parsing of documents in a corpus, reading class label files, etc. We highlight a few of these key functions here. See the full documentation for more details.

### Dependency Parsing and Semantic Role Labeling

The ```SRL``` function of the **Util** module runs the _ClearNLP_ parser to extract both dependency pairs and semantic role labels in all of the files in a specified directory. For example, if I had a data set called ```my_data_set``` in the ```data_sets``` folder. I would run the parser on both my training and testing data in the following way.

```Python
import Util
Util.SRL('<Path-To>/data_sets/my_data_set/train'):
Util.SRL('<Path-To>/data_sets/my_data_set/test'):
```

The parser will produce a ```.srl``` file for every file in ```train``` and ```test```, as shown in the above description of datasets. This file contains both the depedency pairs and semantic role information.

**NOTE**. Please first set up the _ClearNLP_ parser as speficied by: http://clearnlp.wikispaces.com/

**NOTE**. Please refer to the _ClearNLP_ documentation for the speficiations of the ```.srl``` file format: http://clearnlp.wikispaces.com/dataFormat

**NOTE**. The parser _WILL_ parse hidden files in the given directories, but the ```SRL``` function will delete the _parsed_ version of these files. 

### Loading Class Label Files

The class labels associated with each document in data set are stored in the ```train_classes.txt``` and ```test_classes.txt```. The _ith_ line of these documents stores, in plain text, the class labels of the document with filename is lexicographically _ith_ in the respective folders. If a document has multiple labels, the labels are store on the same line seperated by white space. The function ```LoadClassFile``` is used to load the class labels. These class labels are stored as numbers (typically 0 based).

```
Y = Util.LoadClassFile('<Path-To>/data_sets/<Data-Set-Name>/train_classes.txt')
Y = Util.LoadClassFile('<Path-To>/data_sets/<Data-Set-Name>/test_classes.txt')
```

In the case where _every_ document only has one label, ```Y``` in the above example is a 1-by-N numpy array of class label numbers, where N is the number of documents. In the case where _one or more_ documents have _at least_ one label, ```Y``` is an N-by-C matrix where ```C``` is the total number of class labels and ```N``` is the number of documents. Each class label corresponds to a column of ```Y```. Each document has a corresponding row in ```Y``` with 0s and 1s in each column representing whether or not the document has that class label. This follows the specification in the ``sklearn`` package for multiclass-multioutput labels. 


# The Features Module
## Configuration

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

## Defining and Extracting Features

The **Features** module provides the method ```Features()``` to define a _feature definition_ and extract _feature vectors_ from a set of documents. First configure your **Features** module as described above. Then, given a training set of documents ```<Path-To>/data_sets/<Data-Set-Name>/train``` we can extract features from all of the ```.srl``` files in ```train``` in the following way:

```
(feature_def, X_train) = Features.Features('<Path-To>/data_sets/<Data-Set-Name>/train')
```

The first output argument ```feature_def``` is a 1-by-D array of the _units_ which correspond to each each of the D columns of ```X_train```. The second output argument is ```X_train``` an N by D matrix where N is the number of documents and D is the dimensionality of the _feature definition_. The _ith_ row of ```X_train``` corresponds to the lexicographically _ith_ document in ```'<Path-To>/data_sets/<Data-Set-Name>/train'```. 

To extract features from another set of documents (e.g. a testing set of documents) using the same _feature definition_ we pass  ```feature_def``` as an argument to the method. This optional argument has the name ```feature```. For example, extracting _feature vectors_ from ```'<Path-To>/data_sets/<Data-Set-Name>/test'```

```Python
X_test = Features.Features('<Path-To>/data_sets/<Data-Set-Name>/train', feature=feature_def)
```

The next section explains how these can be used in supervised and unsupervised learning.


# Supervised Learning

To do supervised learning with the features extracted from documents we provide the ```SupervisedLearning``` module, which acts as a wrapper to some of the classifiers provided by ```sklearn```. 

The easiest way to use ```SupervisedLearning`` is to use its ``Run`` function. The signature for the function is:

```Python
def Run(FeaturesModule, clf, dirname):
```

The inputs are a configured ```Features``` module, the string name of a classifier to use, and the directory of the data set to be used (formatted in the way that is described above). ```clf``` can alternatively be one of the classifier objects of ```sklearn``` (additional documentation for this to come). The possible values for ```clf``` using the string input are: ```'ridge', 'percepton', 'Passive Aggressive', 'LinearSVM', 'SVM', 'SGD'```. 

The output of the function is a tuple containing the following information in this order:

```
Accuracy
Overall Preicision
Overall Recall
Overall F1 score
Avg. Precision per class
Avg. Recall per class
F1 Score
Precision per class
Recall per class
F1 Score per class
```

# Unsupervised Learning

The document clustering or ```UnsupervisedLearning``` module is used in much of the same way as the Supervised module. The signature of the method ```Run``` in this case is:

```Python
def Run(FeaturesModule, clstr, dirname, train_test='train'):
```

The inputs are a configured ```Features``` module, the string name of a clustering algorithm (```KMeans``` or ```GMM```), the directory of the data set and whether the training document or testing documents of the training set are to be used. 

The method returns a tuple with the following information in the order presented below:

```
Purity Score
Normalized Mutual Information Score
Rand Index Score
```
