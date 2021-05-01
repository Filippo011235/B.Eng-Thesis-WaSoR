# example of stratified k-fold cross-validation with an imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from random import shuffle
import numpy as np

# Wadaba objects have indecies 1 .. 100 without 16
X = np.array(range(1,101))
X = np.delete(X, np.argwhere(X == 16))
# Class occurences 
a01 = [1]*55
a02 = [2]*15
a05 = [5]*16
a06 = [6]*13
y = np.array(a01 + a02 + a05 + a06)

shuffle(y)


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# enumerate the splits and summarize the distributions
for train_ix, test_ix in kfold.split(X, y):
    # print("\n -------- Next Fold ---------")
    # print(len(train_ix))
    # print(train_ix)
    # print(len(test_ix))
    # print(test_ix)
	# select rows
    train_X, test_X = X[train_ix], X[test_ix]
    train_y, test_y = y[train_ix], y[test_ix]
    # # summarize train and test composition
    train_0, train_1 = len(train_y[train_y==1]), len(train_y[train_y==2])
    test_0, test_1 = len(test_y[test_y==1]), len(test_y[test_y==2])
    train_2, train_3 = len(train_y[train_y==5]), len(train_y[train_y==6])
    test_2, test_3 = len(test_y[test_y==5]), len(test_y[test_y==6])
    print("\n -------- Next Fold ---------")
    print('>Train: 1=%d, 2=%d, 5=%d, 6=%d' % (train_0, train_1, train_2, train_3))
    print('>Test: 1=%d, 2=%d, 5=%d, 6=%d' % (test_0, test_1, test_2, test_3))
    # print("Test values")
    # print(test_X)
