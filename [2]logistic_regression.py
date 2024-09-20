# Ali Behfarnia
# Created 2018, Editted 09/2024
# Logistic Regression Classifier

# The purpose of this programming assignment is to write simple python codes for training
# a Logistic Regression classifier using cross validation to correctly predict handwritten
# digit 8. A precision-recall curve will be obtained, and from this curve we would select
# a suitable threshold to develop a classifier that ensures a certain level of precision.

# Note: the runtime for the first part is quick but the second part may take a few mins.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from sklearn.datasets import fetch_openml # 3 vs 5 classifier
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y=y.astype('int')
 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
y_train=(y_train==8)
y_test=(y_test==8)

num_pipeline = Pipeline([
('imputer', SimpleImputer(strategy="median")),
('std_scaler', StandardScaler()),
])

X_train_tr = num_pipeline.fit_transform(X_train)
X_test_tr = num_pipeline.fit_transform(X_test)

log_reg = LogisticRegression(max_iter=200)

y_train_pred = cross_val_predict(log_reg, X_train_tr, y_train, cv=3)

cm=confusion_matrix(y_train, y_train_pred)
print('')
print("Confusion Matrix:\n", cm)
print('')

ps=precision_score(y_train, y_train_pred)
print('')
print("Precision scores:", ps)
print('')

rs=recall_score(y_train, y_train_pred)
print('')
print("Recall scores:", rs)
print('')

Y_scores = cross_val_predict(log_reg, X_train_tr, y_train, cv=3, method="predict_proba")

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="best")
    plt.ylim([0.1, 1])
    
precisions, recalls, thresholds = precision_recall_curve(y_train, Y_scores[:,1])
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

Y_train_pred_90 = (Y_scores[:,1] > 0.9)
ps_target=precision_score(y_train, Y_train_pred_90)
print('')
print("Target precision scores:", ps_target)
print('')

rs_target=recall_score(y_train, Y_train_pred_90)
print(' ')
print("Target recall scores:", rs_target)
print(' ')

log_reg.fit(X_train_tr,y_train)
Y_test_prob=log_reg.predict_proba(X_test_tr)
Y_test_pred_90 = (Y_test_prob[:,1] > 0.9)
ps_test=precision_score(y_test, Y_test_pred_90)
print(' ')
print("Test precision scores:", ps_test)
print(' ')

rs_test=recall_score(y_test, Y_test_pred_90)
print(' ')
print("Test recall scores:", rs_test)
print(' ')