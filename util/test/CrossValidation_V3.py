from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
from sklearn import preprocessing
from util.load_datasets import AR_loader
train_UGIP = True
train_GM = True

# Number of random trials
NUM_TRIALS = 1

corpora_UGIP = get_ArgRecognition_UGIP_dataset()
UGIP_target = corpora_UGIP['label'].to_numpy()

UGIP_features = corpora_UGIP.drop('label', axis=1)
UGIP_features = UGIP_features.apply(LabelEncoder().fit_transform).to_numpy()


corpora_GM = get_ArgRecognition_GM_dataset()
GM_target = corpora_GM['label'].to_numpy()

GM_features = corpora_GM.drop('label', axis=1)
GM_features = GM_features.apply(LabelEncoder().fit_transform).to_numpy()


# Set up possible values of parameters to optimize over
p_grid = {"C": [0.1, 0.5, 1, 10, 100],
          "gamma": [.01, .1, 0.5, 1.0]}

# We will use a Support Vector Classifier with "rbf" kernel
svm = SVC(kernel="rbf", gamma='auto')

# Choose cross-validation techniques for the inner and outer loops,
# independently of the dataset.
inner_cv = KFold(n_splits=3, shuffle=True, random_state=6)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=6)
if train_UGIP:
    print("##################### UGIP #######################")
    clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv, iid=False)
    clf.fit(UGIP_features, UGIP_target)
    #best_sore = clf.best_score_
    #print("Best score:", best_sore)
    nested_score = cross_val_score(clf, X=UGIP_features, y=UGIP_target, cv=outer_cv)
    print(nested_score)
    #print("BEst params", clf.best_estimator_)
    print(nested_score.mean())
if train_GM:
    print("##################### GM #######################")
    clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv, iid=False)
    clf.fit(GM_features, GM_target)
    #best_sore = clf.best_score_
    #print("Best score:", best_sore)
    nested_score = cross_val_score(clf, X=GM_features, y=GM_target, cv=outer_cv)
    print(nested_score)
    # print("BEst params", clf.best_estimator_)
    print(nested_score.mean())