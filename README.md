NLPProject
==========

# Dependencies

Numpy and Scipy are required for using our code. We recommend downloading this distribution of Python, which includes all of Numpy and Scipy and many other useful packages: https://store.continuum.io/cshop/anaconda/

You will also need to install the Enum package for Python, which can be done with the command:
```
$easy_install enum
'''
As well as the Genism package, which can be done with the command:
```
$easy_install -U gensim
'''






# Other Notes



Example Usage of Dependency.py

cd NLPProject/Code
import Dependency as DP

Load the dependencies from the first training file

dep_from_file = DP.ReadDependencyParseFile('../data_sets/reuters_21578/train/train_00001.srl')

Display the dependencies so you can see that it worked properly

DP.Display(dep_from_file)

This shows something like:

{published, to} -- Sentence #19
{published, be} -- Sentence #19
{published, by} -- Sentence #19
{published, after} -- Sentence #19
{figures, Final} -- Sentence #19
{figures, for} -- Sentence #19
{ends, which} -- Sentence #19

Load another file's dependencies

dep_from_file2 = DP.ReadDependencyParseFile('../data_sets/reuters_21578/train/train_14091.srl')

Merge the dependencies

from copy import copy
all_dep = copy(dep_from_file)
all_dep = all_dep + dep_from_file2

Define a feature

feature = DP.DefineFeature(all_dep)
DP.Display(feature)

Extract Feature Vectors

feature_doc_1 = DP.ExtractFeature(feature, dep_from_file)
feature_doc_2 = DP.ExtractFeature(feature, dep_from_file)

If you look at what the features looks like
feature_doc_1

array([1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1,
       1, 5, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1,
       1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 3, 3, 1, 1, 1, 0, 3, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 0, 1, 1, 1, 1,
       2, 1, 1, 1, 2, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 0, 1, 1, 1,
       1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 8, 3, 1, 1, 1, 1, 2, 1], dtype=object)
feature_doc_2

array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=object)
