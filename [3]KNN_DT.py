# Ali Behfarnia
# Created 2017, Editted 09/2024
# KNN and DT Classifiers

# Problem Definition:
# Using k-nearest neighbor (KNN) and decision tree (DT) and random forest
# for hand-written digits classification.

# Data:
# The training and testing data: zip.train, zip.test; 
# Row is the number of samples; The first column of train/test files represent class labels;
# Label digits are from {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}. So, we have a 10-class classification.
# We have 7291 rows (samples) and 257 columns (1 label + 256 features (each image 16*16 pixels)).


import pandas as pp
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


tmp=pp.read_csv('zip.train',delim_whitespace=True, skipinitialspace=True,header=None)
train_data=tmp.values
Y_train=train_data[:,0]
X_train=train_data[:,1:]
print(train_data.shape)
#print(X_train.shape)
#print(Y_train.shape)
#print(test_data[3,:])
tmp=pp.read_csv('zip.test',delim_whitespace=True, skipinitialspace=True,header=None)
test_data=tmp.values
Y_test=test_data[:,0]
X_test=test_data[:,1:]
print(test_data.shape)
#print(X_test.shape)
#print(Y_test.shape)

#=================KNN=========================
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,Y_train) 
y_pred=neigh.predict(X_test)

num_error=(Y_test!=y_pred).sum();

percent_error=(num_error/test_data.shape[0])*100
print(" ")
print("Number of mistalebed points for KNN out of total {} points : {}".format(test_data.shape[0],num_error))
print("Percent error = {:.2f}".format(percent_error))
print(" ")
#===================DT==============================
clf = tree.DecisionTreeClassifier(min_samples_leaf=5)
clf = clf.fit(X_train, Y_train)

y_pred=clf.predict(X_test)

num_error=(Y_test!=y_pred).sum();

percent_error=(num_error/test_data.shape[0])*100
print(" ")
print("Number of mistalebed points for DT out of total {} points : {}".format(test_data.shape[0],num_error))
print("Percent error = {:.2f}".format(percent_error))
print(" ")

#===================RF============================

clf = RandomForestClassifier(n_estimators=50)
clf = clf.fit(X_train, Y_train)

y_pred=clf.predict(X_test)

num_error=(Y_test!=y_pred).sum();

percent_error=(num_error/test_data.shape[0])*100

print("Number of mistalebed points for RF out of total {} points : {}".format(test_data.shape[0],num_error))
print("Percent error = {:.2f}".format(percent_error))
