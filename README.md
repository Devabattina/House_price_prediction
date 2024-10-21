#House Price Prediction using Machine Learning in Python:
We have all been in a situation where we must seek out a new house to purchase. Then comes the part where a lot of fraudsters start making their presence felt, followed by negotiations for deals and selecting places around which to buy a house.

#House Price Prediction using Machine Learning:
So to deal the issues we will use MACHINE LEARNING Based model, trained on the House Price Prediction Dataset

Import Libraries and Dataset
Here, we are making use of:
```
Pandas – To import the DataFrame
Matplotlib – To plot the features of the data. That is, in the case of barplot
Seaborn – visualization of the relationship between features by heatmap.
```
As we have imported the data. So shape method will show us the dimension of the dataset. 

#Data PreProcessing:
Now, we categorize the features depending on their datatype (int, float, object) and then calculate the number of them. 

#Exploratory Data Analysis (EDA):
Exploratory Data Analysis (EDA) is the process of analyzing and visualizing data to uncover patterns, relationships, and anomalies. It helps in understanding data distributions, detecting outliers, and guiding feature selection for better modeling and analysis.To analyze the different categorical features. Draw the barplot.

#Data Cleaning
After all, data cleaning is the need to prepare datasets for training models by deleting the wrong or irrelevant information. The most important steps would include the identification of unwanted columns like ID columns and the NULL values will be either deleted or imputed-mean, mode, or zero. That is, at the end of data cleaning, the dataset could prove to be reliable and relevant. Better model performance and accuracy are, therefore, ensured by proper cleaning of data.


Replace blank SalePrice values with the mean so data will be symmetrical as much as possible.Remove all rows that have nulls in them (since they are very few).Verify columns that have some nulls in the new dataframe if there are any remaining. One hot encoding is the best way to encode categoricals into binary vectors. This value translates directly into the integer values. Using OneHotEncoder we can easily convert object data into int. So for that, first of all, we have to collect all the features which have the object datatype. For doing so, we will make a loop.

#Splitting Dataset into Training and Testing
X and Y splitting (i.e. Y is the SalePrice column and the rest of the other columns are X)

Model and Accuracy
We are training the model to recognize the continuous values so we will make use of these regression models.
'''

SVM- Support Vector Machine
Random Forest Regressor
Linear Regressor
'''
And for loss calculation, we will be utilizing the mean_absolute_percentage_error module. It can easily be imported using the sklearn library.

#SVM – Support vector Machine
SVM is one of the more versatile algorithms of machine learning machines, efficient both for linear as well as nonlinear classification, regression, and outlier detection based on the position of the hyperplane that achieves the maximum margin for a class separation. This adaptability allows it to be used in almost any application -especially text and image classification, spam detection, and anomaly detection.

#Linear Regression
One of the fields of artificial intelligence is machine learning, which produces algorithms that learn from experience by making predictions. Supervised learning is one of the most important techniques for regression and classification. Regression is the problem where the value of the function to be predicted falls within a continuum of possible output values, while classification is the problem where there is a space of finite or countably infinite cardinality.

#CatBoost Classifier
CatBoost is an open-source machine learning algorithm that was designed by Yandex. The CatBoost library is pretty intuitive and compatible with several deep learning frameworks: Apple's Core ML and Google's TensorFlow. Thus, the primary advantages of the CatBoost library are performance, ease of use, and robustness.

#Conclusion:
Clearly, SVM model is giving better accuracy as the mean absolute error is the least among all the other regressor models i.e. 0.18 approx.House price prediction provides far more information regarding location and size, and it uses machine learning algorithms that are priceless and make for good decision-making in the real estate market. A better model contributes to a more efficient housing landscape, obviously enhancing investment strategies.
