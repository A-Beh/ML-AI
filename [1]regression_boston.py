# Ali Behfarnia
# Created 2018, Editted 09/2024
# Simple ML Regression

# Problem Definition:
# The purpose of this programming assignment is to write simple python
# codes for training Machine Learning model and predicting housing prices.
# The data contains the owner-occupied home prices in the Boston area during 1970s. 
# The dataset has 506 rows and 14 columns. The last column contains the median value 
# of owner-occupied homes in $1000’s (this is the response variable y for regression 
# problem). The first 13 columns represent various attributes /features, for example, 
# per capita crime rate by town, average number of rooms per dwelling, index of 
# accessibility to radial highways, pupil-teacher ratio by town etc.



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

housing_data=pd.read_csv('boston_housing.data',delimiter=' ',header=None)
m,n=housing_data.shape
housing_data.iloc[:,n-1]=housing_data.iloc[:,n-1]*1000
#print(housing_data[n-1])

train_set, test_set = train_test_split(housing_data, test_size=0.3, random_state=42)

# print(train_set.shape)

X_train = train_set.drop(n-1,axis=1)
X_test = test_set.drop(n-1,axis=1)
Y_train = train_set.iloc[:,n-1]
Y_test = test_set.iloc[:,n-1]

# scaling
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

X_train_set_tr = num_pipeline.fit_transform(X_train)
X_test_set_tr = num_pipeline.fit_transform(X_test)

def display_scores(scores):
	print("Scores:", scores)
	print("Mean:", scores.mean())
	print("Standard deviation:", scores.std())
    
#linear regression    
lin_reg = LinearRegression()
lin_reg.fit(X_train_set_tr,Y_train)
lin_scores = cross_val_score(lin_reg, X_train_set_tr, Y_train, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("                            ")
print("***Linear regression***")
print("                            ")
display_scores(lin_rmse_scores)

# Decision Tree Regressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train_set_tr,Y_train)
tree_scores = cross_val_score(tree_reg, X_train_set_tr, Y_train,
scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
print("                            ")
print("***DecisionTree Regressor***")
print("                            ")
display_scores(tree_rmse_scores)


#RandomForest Regressoe
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train_set_tr,Y_train)
forest_scores = cross_val_score(forest_reg, X_train_set_tr, Y_train, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print("                            ")
print("***RandomForest Regressor***")
print("                            ")
display_scores(forest_rmse_scores)

#SVM Regressor
svm_reg = SVR(kernel='rbf',gamma=0.01)
svm_reg.fit(X_train_set_tr, Y_train)
svm_scores = cross_val_score(svm_reg, X_train_set_tr, Y_train,
scoring="neg_mean_squared_error", cv=10)
svm_rmse_scores = np.sqrt(-svm_scores)
print("                            ")
print("***SVM Regressor***")
print("                            ")
display_scores(svm_rmse_scores)


param_grid = [ {'n_estimators': [10, 20, 30, 40], 'max_features': [4,8]},]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=10,
scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(X_train_set_tr, Y_train)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score), params)

print(grid_search.best_params_)
final_model = grid_search.best_estimator_

# prediction on test data
X_test_set_tr = num_pipeline.fit_transform(X_test)

forest_test_predictions = final_model.predict(X_test_set_tr)
forest_test_mse = mean_squared_error(Y_test, forest_test_predictions)
forest_test_rmse = np.sqrt(forest_test_mse)
print("                            ")
print("                            ")
print('Test error of RandomForest regressor is {}.'.format(forest_test_rmse))
