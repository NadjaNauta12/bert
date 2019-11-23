import operator
import numpy as np
from sklearn import cross_validation
from sklearn import ensemble
from sklearn.datasets import load_boston

# set random state
state = 1

# load boston dataset
boston = load_boston()

X = boston.data
y = boston.target

outer_scores = []

# outer cross-validation
outer = cross_validation.KFold(len(y), n_folds=3, shuffle=True, random_state=state)
for fold, (train_index_outer, test_index_outer) in enumerate(outer):
    X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
    y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]

    inner_mean_scores = []

    # define explored parameter space.
    # procedure below should be equal to GridSearchCV
    tuned_parameter = [1000, 1100, 1200]
    for param in tuned_parameter:

        inner_scores = []

        # inner cross-validation
        inner = cross_validation.KFold(len(X_train_outer), n_folds=3, shuffle=True, random_state=state)
        for train_index_inner, test_index_inner in inner:
            # split the training data of outer CV
            X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
            y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]

            # fit extremely randomized trees regressor to training data of inner CV
            clf = ensemble.ExtraTreesRegressor(param, n_jobs=-1, random_state=1)
            clf.fit(X_train_inner, y_train_inner)
            inner_scores.append(clf.score(X_test_inner, y_test_inner))

        # calculate mean score for inner folds
        inner_mean_scores.append(np.mean(inner_scores))

    # get maximum score index
    index, value = max(enumerate(inner_mean_scores), key=operator.itemgetter(1))

    print ('Best parameter of %i fold: %i' % (fold + 1, tuned_parameter[index]))

    # fit the selected model to the training set of outer CV
    # for prediction error estimation
    clf2 = ensemble.ExtraTreesRegressor(tuned_parameter[index], n_jobs=-1, random_state=1)
    clf2.fit(X_train_outer, y_train_outer)
    outer_scores.append(clf2.score(X_test_outer, y_test_outer))

# show the prediction error estimate produced by nested CV
print ('Unbiased prediction error: %.4f' % (np.mean(outer_scores)))

# finally, fit the selected model to the whole dataset
clf3 = ensemble.ExtraTreesRegressor(tuned_parameter[index], n_jobs=-1, random_state=1)
clf3.fit(X, y)