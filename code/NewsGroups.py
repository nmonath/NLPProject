from __future__ import print_function
import os
from sklearn.datasets import fetch_20newsgroups
import Util


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
  data_train = fetch_20newsgroups(subset='train', categories=None, shuffle=True, random_state=42)
  data_test = fetch_20newsgroups(subset='test', categories=None, shuffle=True, random_state=42)

  if dirname[-1] == '/' or dirname[-1] == '\\':
    dirname = dirname[:-1]
  
  Util.WriteClassFile(data_train.target, os.path.join(dirname, 'train_classes.txt'))
  Util.WriteClassFile(data_test.target,os.path.join(dirname, 'test_classes.txt'))


  train_counter = 0;
  for doc in data_train.data:
    filename = 'train_' + str(train_counter).zfill(5);
    f = file(os.path.join(dirname, 'train', filename), 'w');
    f.write(doc.encode('ascii', 'ignore'));
    f.close();
    train_counter = train_counter + 1;

  test_counter = 0;
  for doc in data_test.data:
    filename = 'test_' + str(test_counter).zfill(5);
    f = file(os.path.join(dirname, 'test', filename), 'w');
    f.write(doc.encode('ascii', 'ignore'));
    f.close();
    test_counter = test_counter + 1;

  class_index = file(os.path.join(dirname, 'class_label_index.txt'), 'w')
  for label in data_train.target_names:
    class_index.write(label + '\n')
  class_index.close()



