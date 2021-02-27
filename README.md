# Health Insurance Cost Prediction - Reproducibility Analysis 
Determining the reproducibility of the Shinde A. et al 2020 paper "Comparative Study of Regression Models and Deep Learning Models for Insurance Cost Prediction" published in Springer
by implementing the same dataset and regression methods. 

## General Info
The aim of the paper was to predict insurance costs using various regression methods and comparing their results. 

This project examines the methods used in the cited paper and attempts to recreate the results using the original dataset. 

Implementations in this project:
- Multiple Linear Regression
- Support Vector Regression
- Random Forest Regressor
- XGBoost

##Technologies
* Python version:       3.9
* Pandas version:       1.2.2
* Xgboost version:      1.3.3
* Scikit-learn version: 0.24.1

##Dataset
The dataset “Medical Cost Personal Dataset” comes from https://www.kaggle.com/mirichoi0218/insurance

Includes attributes:
Age
Body Mass Index (BMI)
Number of Children
Region Location (Southwest, Southeast, Northwest, Northeast)
Sex
Smoker (Whether someone currently smokes Y/N)
Charges (Target variable)



##Citations
Kaggle Medical Cost Personal Datasets. Kaggle Inc. https://www.kaggle.com/mirichoi0218/insurance

Shinde A., Raut P. (2020) Comparative Study of Regression Models and Deep Learning Models for Insurance Cost Prediction. In: Abraham A., Cherukuri A., Melin P., Gandhi N. (eds) Intelligent Systems Design and Applications. ISDA 2018 2018. Advances in Intelligent Systems and Computing, vol 940. Springer, Cham. https://doi.org/10.1007/978-3-030-16657-1_103
