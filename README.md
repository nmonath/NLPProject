NLPProject
==========

# Dependencies

Numpy and Scipy are required for using our code. We recommend downloading this distribution of Python, which includes all of Numpy and Scipy and many other useful packages: https://store.continuum.io/cshop/anaconda/

You will also need to install the Enum package for Python, which can be done with the command:
```
$ easy_install enum
```
As well as the Genism package, which can be done with the command:
```
$ easy_install -U gensim
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
data_set_path = '<PATH_TO>/data_set/<DATA_SET_NAME>/'
(feature_definition, x_train) = Features.Features(data_set_path)
```
