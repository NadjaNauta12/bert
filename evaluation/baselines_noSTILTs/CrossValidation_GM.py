"""
Adapted from https://stats.stackexchange.com/questions/136296/implementation-of-nested-cross-validation

Using this information, we can then perform a separate K-fold CV loop for parameter tuning of the selected models.
"""
import itertools
import operator
import sys

import numpy as np
from sklearn import ensemble
from sklearn import model_selection
from sklearn.datasets import load_boston

sys.path.append("C:/Users/Wifo/PycharmProjects/Masterthesis")
sys.path.append("/work/nseemann")
sys.path.append("/content/bert")
from load_datasets import AR_loader
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# set random state



state = 6
para_c = [0.01, .01, .1, 1, 50]
# NOTE from Seemann :  Gamma has no effect - thas Why I only changes Parameter C
#para_gamma = ['auto', 'scale',   0.0001, .01, .1]
para_gamma = ['auto']
configs = list(itertools.product(para_c, para_gamma))

print(" Cross Validation with Setting - GM_ONLY")
corpora_GM = AR_loader.get_ArgRecognition_dataset(2)[2]
GM_target = corpora_GM['label'].to_numpy()
GM_features = corpora_GM.drop('label', axis=1)
GM_features = GM_features.apply(LabelEncoder().fit_transform).to_numpy()


outer_scores = []
# outer cross-validation
#outer = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=state) # TODO test
outer = model_selection.KFold(n_splits=5, shuffle=True, random_state=state)
fold_counter = 0

for train_index_outer, test_index_outer in outer.split(corpora_GM):#range(0,len(X))):
    print("Train_IDX", train_index_outer.shape, train_index_outer[:10])
    print("Test_IDX", test_index_outer.shape, test_index_outer[:10])
    X_train_outer, X_test_outer = GM_features[train_index_outer], GM_features[test_index_outer]
    y_train_outer, y_test_outer = GM_target[train_index_outer], GM_target[test_index_outer]

    inner_mean_scores = []

    # define explored parameter space - equal to GridSearchCV
    for (c, gamma) in configs:
        print("Settings: ", "C=", c, "Gamma=", gamma)
        cv_mean_scores = []

        # inner cross-validation
        # inner = model_selection.KFold(len(X_train_outer), n_folds=3, shuffle=True, random_state=state)
        folds = model_selection.KFold(n_splits=3, shuffle=True, random_state=state + 1)
        for train_idx, test_idx in folds.split(X_train_outer):
            print("Train_IDX inner", train_idx.shape, train_idx[:10])
            print("Test_IDX inner", test_idx.shape, test_idx[:10])
            print()
            # split the training data of outer CV
            X_train, X_test = GM_features[train_idx], GM_features[test_idx]
            y_train, y_test = GM_target[train_idx], GM_target[test_idx]

            # fit extremely randomized trees regressor to training data of inner CV
            #clf = ensemble.ExtraTreesRegressor(param, n_jobs=-1, random_state=1)
            clf = SVC(kernel='rbf', gamma=gamma, C=c, random_state=state)
            #print(clf)
            clf.fit(X_train, y_train)
            print("Score", clf.score(X_test, y_test))
            cv_mean_scores.append(clf.score(X_test, y_test))

        #print("Scores each:", inner_scores)
        # calculate mean score for inner folds
        inner_mean_scores.append(np.mean(cv_mean_scores))
        #inner_mean_scores.append(np.max(inner_scores))
        #print("MAX=", np.max(inner_scores))

    # get maximum score index
    index, value = max(enumerate(inner_mean_scores), key=operator.itemgetter(1))
    #print("Best", index, value)
    print('Best parameter of %i fold: with parasetting %i' % (fold_counter+1,  index))

    # fit the selected model to the training set of outer CV - for prediction error estimation
    clf2 = SVC(C=para_c[index], gamma=para_gamma[index], kernel='rbf')
    clf2.fit(X_train_outer, y_train_outer)
    outer_scores.append(clf2.score(X_test_outer, y_test_outer))
    fold_counter += 1
# show the prediction error estimate produced by nested CV
print("Scores outer loop", outer_scores)
print('Unbiased prediction error: %.4f' % (np.mean(outer_scores)))

mean_scores = []

for (c, gamma) in configs:
    print("Settings: ", "C=", c, "Gamma=", gamma)
    cv_mean_scores = []

    # regular cross-validation
    folds = model_selection.KFold(n_splits=5, shuffle=True, random_state=state + 2) # TODO test STratified

    for train_idx, test_idx in folds.split(corpora_GM):
        print("Train_IDX", train_idx.shape, train_idx[:10])
        print("Test_IDX", test_idx.shape, test_idx[:10])
        print()
        # split the training data of outer CV
        X_train, X_test = GM_features[train_idx], GM_features[test_idx]
        y_train, y_test = GM_target[train_idx], GM_target[test_idx]

        # fit extremely randomized trees regressor to training data of inner CV
        # clf = ensemble.ExtraTreesRegressor(param, n_jobs=-1, random_state=1)
        clf = SVC(kernel='rbf', gamma=gamma, C=c, random_state=state)
        clf.fit(X_train, y_train)
        print("Score", clf.score(X_test, y_test))
        cv_mean_scores.append(clf.score(X_test, y_test))
    mean_scores.append(np.mean(cv_mean_scores))

# get maximum score index
index, value = max(enumerate(mean_scores), key=operator.itemgetter(1))

print( 'Best parameter : %i' % (para_c[index]))