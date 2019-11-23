from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np
from sklearn import preprocessing
from load_datasets.data_loader import get_ArgRecognition_UGIP_dataset, get_ArgRecognition_GM_dataset

print(__doc__)

# Number of random trials
NUM_TRIALS = 1

# Load the dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
print(type(X_iris))
print(type(y_iris))

corpora_UGIP = get_ArgRecognition_UGIP_dataset()
UGIP_target = corpora_UGIP['label'].to_numpy()

UGIP_features = corpora_UGIP.drop('label', axis=1)
UGIP_features = UGIP_features.apply(LabelEncoder().fit_transform).to_numpy()


corpora_GM = get_ArgRecognition_GM_dataset()
GM_target = corpora_GM['label'].to_numpy()

GM_features = corpora_GM.drop('label', axis=1)
GM_features = GM_features.apply(LabelEncoder().fit_transform).to_numpy()



# Set up possible values of parameters to optimize over
p_grid = {"C": [0.1, 1, 10, 100],
          "gamma": [.01, .1, 1.0]}

# We will use a Support Vector Classifier with "rbf" kernel
svm = SVC(kernel="rbf", gamma='auto')

# Choose cross-validation techniques for the inner and outer loops,
# independently of the dataset.
inner_cv = KFold(n_splits=3, shuffle=True, random_state=6)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=6)

clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv, iid=False)
nested_score = cross_val_score(clf, X=UGIP_features, y=UGIP_target, cv=outer_cv)
#print("BEst params", clf.best_estimator_)
print(nested_score.mean())

clf.fit(GM_features, GM_target)

y_true, y_pred =GM_target, clf.predict(GM_features)
print(classification_report(y_true, y_pred))

