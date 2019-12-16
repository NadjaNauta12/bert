"""
Adapted from https://stats.stackexchange.com/questions/136296/implementation-of-nested-cross-validation
"""
import itertools
import operator
import sys
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import model_selection
from sklearn.datasets import load_boston

sys.path.append("C:/Users/Wifo/PycharmProjects/Masterthesis")
from load_datasets import AR_loader
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# set random state
state = 6
p_grid = {"C": [00.0001, 0.001, .01, .1],
          "gamma": [0.0001, .001, .01, .1]}
para_c = [0.01, .01, .1]
para_gamma = ['auto',  0.001, .01, .1]
configs = list(itertools.product(para_c, para_gamma))



print (" Cross Validation with Setting 2 - GM_UGIP")
print ("TODO - Check if this is correct?")
# corpora_UGIP = AR_loader.get_ArgRecognition_dataset(1)[2]
# UGIP_target = corpora_UGIP['label'].to_numpy()
#
# UGIP_features = corpora_UGIP.drop('label', axis=1)
# UGIP_features = UGIP_features.apply(LabelEncoder().fit_transform).to_numpy()
#
#
# corpora_GM = AR_loader.get_ArgRecognition_dataset(2)[2]
# GM_target = corpora_GM['label'].to_numpy()
#
# GM_features = corpora_GM.drop('label', axis=1)
# GM_features = GM_features.apply(LabelEncoder().fit_transform).to_numpy()
#
# corpora_UGIP = AR_loader.get_ArgRecognition_dataset(1)[2]
# corpora_GM = AR_loader.get_ArgRecognition_dataset(2)[2]
# corpora = pd.concat((corpora_GM, corpora_UGIP), axis=0)
#
# target = corpora['label'].to_numpy()
# features = corpora.drop('label', axis=1)
# features = features.apply(LabelEncoder().fit_transform).to_numpy()
#
# outer_scores = []
#
# # outer cross-validation
# outer = model_selection.KFold(n_splits=5, shuffle=True, random_state=state)
# fold_counter =0
# for train_index_outer, test_index_outer in outer.split(corpora):#range(0,len(X))):
#     X_train_outer, X_test_outer = features[train_index_outer], features[test_index_outer]
#     y_train_outer, y_test_outer = target[train_index_outer], target[test_index_outer]
#
#     inner_mean_scores = []
#
#     # define explored parameter space.
#     # procedure below should be equal to GridSearchCV
#     #tuned_parameter = [1000, 1100, 1200]
#     #configs = list(itertools.product(train_batch_sizes_list, learning_rate_list, train_epochs_list))
#
#     #for (train_batch_size, learning_rate, train_epochs) in configs:
#     for (c, gamma) in configs:
#         print("Settings: ", "C=", c, "Gamma=", gamma)
#         inner_scores = []
#
#         # inner cross-validation
#         # inner = model_selection.KFold(len(X_train_outer), n_folds=3, shuffle=True, random_state=state)
#         inner = model_selection.KFold(n_splits=3, shuffle=True, random_state=state+1)
#         for train_index_inner, test_index_inner in inner.split(corpora):
#             # split the training data of outer CV
#             X_train_inner, X_test_inner = features[train_index_inner], features[test_index_inner]
#             y_train_inner, y_test_inner = target[train_index_inner], target[test_index_inner]
#
#             # fit extremely randomized trees regressor to training data of inner CV
#             #clf = ensemble.ExtraTreesRegressor(param, n_jobs=-1, random_state=1)
#             clf = SVC(kernel='rbf', gamma=gamma, C=c, random_state=state)
#             print(clf)
#             clf.fit(X_train_inner, y_train_inner)
#             inner_scores.append(clf.score(X_test_inner, y_test_inner))
#
#         print("Scores each:", inner_scores)
#         # calculate mean score for inner folds
#         inner_mean_scores.append(np.mean(inner_scores))
#         #inner_mean_scores.append(np.max(inner_scores))
#         #print("MAX=", np.max(inner_scores))
#
#     # get maximum score index
#     index, value = max(enumerate(inner_mean_scores), key=operator.itemgetter(1))
#     #print("Best", index, value)
#     print ('Best parameter of %i fold: with parasetting %i' % (fold_counter+1,  index))
#
#     # fit the selected model to the training set of outer CV
#     # for prediction error estimation
#     clf2 = SVC(C=para_c[index], gamma=para_gamma[index], kernel='rbf')
#     clf2.fit(X_train_outer, y_train_outer)
#     outer_scores.append(clf2.score(X_test_outer, y_test_outer))
#     fold_counter += 1
# # show the prediction error estimate produced by nested CV
# print(outer_scores)
# print('Unbiased prediction error: %.4f' % (np.mean(outer_scores)))
#
# # finally, fit the selected model to the whole dataset
# clf3 = SVC(C=para_c[index], gamma=para_gamma[index], random_state=state+3)
# clf3.fit(features, target)
