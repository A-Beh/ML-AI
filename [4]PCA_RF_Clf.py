# Ali Behfarnia
# Editted 09/2024
# Study the Impact of PCA on a classifcation problem
# Metric: ROC_AUC

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



X, Y = load_digits(return_X_y=True)

print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state =  42)

y_train_85 = ((y_train == 8) | (y_train == 5))
y_test_85 =  ((y_test == 8) | (y_test == 5))

num_pip = Pipeline([ 
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scl', StandardScaler())
])

X_train_tr = num_pip.fit_transform(X_train)
X_test_tr = num_pip.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

RF_clf = RandomForestClassifier()
para_grid = [{'n_estimators':[100, 200, 300, 400]}]
grid_clf = GridSearchCV(RF_clf, para_grid, cv = 3, scoring='roc_auc')
grid_clf.fit(X_train_tr, y_train_85)
cvres = grid_clf.cv_results_
# print(cvres)

for mean_score, para in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, para)

print(grid_clf.best_estimator_)

final_model = grid_clf.best_estimator_
final_model.fit(X_train_tr, y_train_85)
pred = final_model.predict_proba(X_test_tr)[:,1]

# print(pred.shape)
# print(y_test.shape)
auc_score = roc_auc_score(y_test_85, pred)
print(f"The ROC_AUC score is {auc_score:.3f}.")
print('******************************************\n')

from sklearn.decomposition import PCA

pca = PCA()
X_train_reduced = pca.fit_transform(X_train_tr)

print(X_train_tr.shape)
print(X_train_reduced.shape)

d_list = []
pca_auc_list = [] 
for i in range(0,32,2):
    i +=2
    X_train_reduced_i = X_train_reduced[:,0:i]
    RF_clf = RandomForestClassifier()
    grid_clf = GridSearchCV(RF_clf, para_grid, cv = 3, scoring='roc_auc')
    grid_clf.fit(X_train_reduced_i, y_train_85)
    final_model = grid_clf.best_estimator_
    final_model.fit(X_train_reduced_i,y_train_85)
    pca_vectors=pca.components_.T[:,0:i]
    X_test_reduced=X_test_tr.dot(pca_vectors)
    pred_i = final_model.predict_proba(X_test_reduced)[:,1]
    Y_test_score = roc_auc_score(y_test_85, pred_i)
    d_list.append(i)
    pca_auc_list.append(Y_test_score)
    print('Test roc_auc score with d={} principal components is {}.'.format(i,Y_test_score))


import matplotlib.pyplot as plt
plt.plot(d_list, pca_auc_list)
plt.xlabel("The number of Principle components")
plt.ylabel("ROC-AUC")
plt.show()

