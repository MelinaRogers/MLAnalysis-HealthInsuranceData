# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgboost
from pip._vendor.colorama import Fore
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

# Training_Dataset = pd.read_csv("train.csv")
# Training_Dataset = Training_Dataset.dropna()
# X_train = np.array(Training_Dataset.iloc[:, :-1].values) # Independent Variable
# y_train = np.array(Training_Dataset.iloc[:, 1].values) # Dependent Variable
#
# Testing_Dataset = pd.read_csv("test.csv")
# Testing_Dataset = Testing_Dataset.dropna()
# X_test = np.array(Testing_Dataset.iloc[:, :-1].values) # Independent Variable
# y_test = np.array(Testing_Dataset.iloc[:, 1].values) # Dependent Variable

# Training the Model for Linear
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
#
# accuracy = regressor.score(X_test, y_test)
# print('Accuracy = '+ str(accuracy))
#
# plt.style.use('seaborn')
# plt.scatter(X_test, y_test, color = 'red', marker = 'o', s = 35, alpha = 0.5,
#           label = 'Test data')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue', label='Model Plot')
# plt.title('Predicted Values vs Inputs')
# plt.xlabel('Inputs')
# plt.ylabel('Predicted Values')
# plt.legend(loc = 'upper left')
# plt.show()

dataset = pd.read_csv('insurance.csv')


dataset.charges = np.log1p(dataset.charges)

X = dataset.iloc[:, :-1] # Independent Variable
y = dataset.iloc[:, -1] # Dependent Variable

# Label Encoding:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X.iloc[:, 1] = le.fit_transform(X.iloc[:, 1])
X.iloc[:, 4] = le.fit_transform(X.iloc[:, 4])

# OneHot Encoding:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# # Training the Model
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
#
# accuracy = regressor.score(X_test, y_test)
# print('Accuracy = '+ str(accuracy))

forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
forest.fit(X_train,y_train)
forest_train_pred = forest.predict(X_train)
forest_test_pred = forest.predict(X_test)

print('MSE train data: %.5f, MSE test data: %.5f' % (
mean_squared_error(y_train,forest_train_pred),
mean_squared_error(y_test,forest_test_pred)))
print('R2 train data: %.5f, R2 test data: %.5f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred)))

xgb_reg = xgboost.XGBRegressor(max_depth=2, random_state=0)
xgb_reg.fit(X_train,y_train)
xgb_test = xgb_reg.predict(X_test)
print(Fore.GREEN + "Accuracy of XGB Regression is : ",xgb_reg.score(X_test,y_test))

rmse = np.sqrt(mean_squared_error(y_test, xgb_test))
print("XGBOOST RMSE: %f" % (rmse))

# xgb.plot_importance(xg_reg)
# plt.rcParams['figure.figsize'] = [5, 5]
# plt.show()