# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgboost
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer

# Dataset to use
dataset = pd.read_csv('insurance.csv')

# Log transformation
dataset.charges = np.log1p(dataset.charges)

X = dataset.iloc[:, :-1]  # Independent Variable
y = dataset.iloc[:, -1]  # Dependent Variable

# Label Encoding:
le = LabelEncoder()
X.iloc[:, 1] = le.fit_transform(X.iloc[:, 1])
X.iloc[:, 4] = le.fit_transform(X.iloc[:, 4])

# OneHot Encoding:
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# 80-20 split of the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()                                     # X scaler
sc_y = StandardScaler()                                     # Y scaler
X_scaled = sc_X.fit_transform(X)                            # Scaling X
y_scaled = sc_y.fit_transform(y.values.reshape(-1, 1))      # Scaling y

# Multiple Linear Regression -------------------------------------------
scaler = StandardScaler()
sc_x_train = scaler.fit_transform(X_train)
sc_x_test = scaler.transform(X_test)
lr = LinearRegression()
lr.fit(sc_x_train, y_train)
mlp_predict = lr.predict(sc_x_test)

mlp_mse = mean_squared_error(y_test,mlp_predict)
mlp_r2 = r2_score(y_test, mlp_predict)
# print("MLP MSE: %.5f" % (mean_squared_error(y_test,mlp_predict )))
# print("MLP R2: %.5f" % (r2_score(y_test, mlp_predict )))

# Random forest initial model ------------------------------------------
forest = RandomForestRegressor(n_estimators=100,
                               criterion='mse',
                               random_state=0,              # random state not put in original paper
                               n_jobs=-1)
forest.fit(X_train, y_train)                                # Fit the points
forest_train_pred = forest.predict(X_train)                 # Predict training
forest_test_pred = forest.predict(X_test)                   # Predict test

rf_mse_train = mean_squared_error(y_train, forest_train_pred)
rf_mse_test = mean_squared_error(y_test, forest_test_pred)
rf_r2_train =  r2_score(y_train, forest_train_pred)
rf_r2_test = r2_score(y_test, forest_test_pred)
# print('Random Forest MSE train data: %.5f, Random Forest MSE test data: %.5f' % (
#     mean_squared_error(y_train, forest_train_pred),
#     mean_squared_error(y_test, forest_test_pred)))
# print('Random Forest R2 train data: %.5f, Random Forest R2 test data: %.5f' % (
#     r2_score(y_train, forest_train_pred),
#     r2_score(y_test, forest_test_pred)))

# XGBoost Model ---------------------------------------------------------
xgb_reg = xgboost.XGBRegressor(max_depth=2, random_state=0)  # XGBOOST regressor
xgb_reg.fit(X_train, y_train)                                # Fit training data
xgb_test = xgb_reg.predict(X_test)                           # Predict with test data

xg_rmse = np.sqrt(mean_squared_error(y_test, xgb_test))
xg_r2 = xgb_reg.score(X_test, y_test)
# print("XGB Regression R2 : ", xgb_reg.score(X_test, y_test))
# print("XGBOOST RMSE: %f" % (xg_rmse))

# Support Vector Regression Model -----------------------------------------


# Creating the SVR regressor
regressor_svr = SVR()

# Applying Grid Search to find the best model/parameters
parameters = {'kernel': ['rbf', 'sigmoid'],
              'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
              'tol': [0.0001],
              'C': [0.001, 0.01, 0.1, 1, 10, 100]}

regressor_svr = GridSearchCV(estimator=regressor_svr,
                             param_grid=parameters,
                             cv=10,
                             #verbose=4,
                             n_jobs=-1)

regressor_svr = regressor_svr.fit(X_scaled, y_scaled.ravel())


# Predicting Cross Validation Score
cv_svr = regressor_svr.best_score_

# Predicting R2 Score the Train set results
y_pred_svr_train = sc_y.inverse_transform(regressor_svr.predict(sc_X.transform(X_train)))
r2_score_svr_train = r2_score(y_train, y_pred_svr_train)

# Predicting R2 Score the Test set results
y_pred_svr_test = sc_y.inverse_transform(regressor_svr.predict(sc_X.transform(X_test)))
r2_score_svr_test = r2_score(y_test, y_pred_svr_test)

# # Predicting RMSE Test set results
rmse_svr = (np.sqrt(mean_squared_error(y_test, y_pred_svr_test)))
# print('SVR CV: ', cv_svr.mean())
# print('SVR R2_score (train): ', r2_score_svr_train)
# print('SVR R2_score (test): ', r2_score_svr_test)
# print("SVR RMSE: \n", rmse_svr)


# RF Try 2 -----------------------------------------------------------------

# Creating the Random Forest regressor

regressor_rf = RandomForestRegressor()
parameters = {"n_estimators": [1200],
              "max_features": ["auto"],
              "max_depth": [50],
              "min_samples_split": [7],
              "min_samples_leaf": [10],
              "bootstrap": [True],
              "criterion": ["mse"],
              "random_state": [0]}

regressor_rf = GridSearchCV(estimator=regressor_rf,
                            param_grid=parameters,
                            cv=10,
                            # verbose = 4,
                            n_jobs=-1)

regressor_rf = regressor_rf.fit(X_scaled, y.ravel())

# Predicting Cross Validation Score
cv_rf = regressor_rf.best_score_

# Predicting R2 Score the Train set results
y_pred_rf_train = regressor_rf.predict(sc_X.transform(X_train))
r2_score_rf_train = r2_score(y_train, y_pred_rf_train)

# Predicting R2 Score the Test set results
y_pred_rf_test = regressor_rf.predict(sc_X.transform(X_test))
r2_score_rf_test = r2_score(y_test, y_pred_rf_test)

# Predicting RMSE the Test set results
rmse_rf = (np.sqrt(mean_squared_error(y_test, y_pred_rf_test)))
# print("Random Forest Results \n")
# print('Random forest CV: ', cv_rf.mean())
# print('Random forest R2_score (train): ', r2_score_rf_train)
# print('Random forest R2_score (test): ', r2_score_rf_test)
# print("Random forest RMSE: ", rmse_rf)



# PRINTING -----------------------
print("\n")
print("---Multiple Linear Regression Results---")
print("MLP MSE: %.5f" % mlp_mse)
print("MLP R2: %.5f" % mlp_r2)
print("\n")
print("---Random Forest Regression Results---")
print('Random Forest MSE: %.5f' % rf_mse_test)
print('Random Forest R2: %.5f' % rf_r2_test)
print("\n")
print("RF second")
print('Random forest CV: ', cv_rf.mean())
print('SVR R2_score (train): ', r2_score_svr_train)
print('Random forest R2_score (test): ', r2_score_rf_test)
print("Random forest RMSE: ", rmse_rf)
print("\n")
print("---Support Vector Regression Results---")
print('SVR CV: ', cv_svr.mean())
print('SVR R2_score (train): ', r2_score_svr_train)
print('SVR R2_score (test): ', r2_score_svr_test)
print("SVR RMSE: ", rmse_svr)
print("\n")
print("---XGBoost Results---")
print("XGB Regression R2 : ", xg_r2)
print("XGBOOST RMSE: %f" % xg_rmse)