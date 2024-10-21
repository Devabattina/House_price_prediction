# House_price_prediction

#Overview
The house price forecasting is done using machine learning models based on different features related to houses like lot area, zoning classification, building type, year built, and so on. The models used for this purpose are as follows:
  *Support Vector Regressor (SVR)
  *Random Forest Regressor
  *CatBoost Regressor
This project thus basically follows a general workflow of any typical machine learning model-loading data, preprocessing data, training, evaluation, and prediction.
  *CatBoost gives the highest accuracy with MAPE 0.1329.
  *Random Forest is the second best with a MAPE of 0.1415.
  *Least optimal is the performance by SVR with a MAPE of 0.3597.
  
#Dataset
This dataset is related to house sales records. Features considered are as given below:
MSSubClass: Type of dwelling
MSZoning: Zoning classification
LotArea: Lot size in square feet
BldgType: Type of building
OverallCond: Overall condition of the house
YearBuilt: Original construction year
TotalBsmtSF: Total square footage of the basement
SalePrice: The target variable that represents the final price of the house.
The data set used is in the folder of data/ or alternatively, you can use any custom data by replacing the script accordingly.

#Data Preprocessing
1. Data Load: Loads the dataset used into a pandas DataFrame.
2. Handling missing values: Drops the missing values in the target variable, SalePrice.
3. Feature Engineering: Picks features to predict house prices while other features are discarded.
4. Scaling: The numerical features are scaled such that their scale would be comparable for the model to do well.

#Models and Training
The project encompasses the following machine learning models for predicting prices:
1. Support Vector Regressor (SVR): This is a kernel-based machine learning algorithm, which can be used for regression-oriented tasks.
2. Random Forest Regressor: It's an ensemble learning method that builds multiple decision trees and combines them to boost accuracy.
3. CatBoost Regressor: Gradient boosting algorithm that is sensitive to categorical features out-of-the-box. Very high precision.
Training procedure:
>>Choose the range for splitting the data between training and validation sets.
>>Fit all the models on the training data
>>Performance of the model will be measured on the validation set.
Hyperparameter options for all the models are natively tunable in the code.
#Performance metrics
We measure the performance of the models in terms of Mean Absolute Percentage Error (MAPE) in predicting the accuracy. MAPE provides an intuitively understandable measure since it calculates the percentage error between the predicted values and actual values.

from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.10f}")

Performance summary of models on the validation set:
SVR MAPE: 0.3597
Random Forest MAPE: 0.1415
CatBoost MAPE: 0.1329

The model that has the least prediction error is CatBoost, so it's the best for this task.



#Results
The performance of each model is output by the project as MAPE. Based on the results:

1. In the case of CatBoost, accuracy is at a much higher side with a MAPE of 0.1329.
2. Random Forest comes second with a MAPE of 0.1415.
3. SVR is the least optimally working with a MAPE of 0.3597.
