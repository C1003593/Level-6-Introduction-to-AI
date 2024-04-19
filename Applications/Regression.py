import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt

#https://www.kaggle.com/datasets/sukhmandeepsinghbrar/heart-attack-dataset

#Load the medical dataset
medical_data = pd.read_csv("Medicaldataset.csv")

#Sort the data into appropriate columns
medical_data.columns = [
    "Age",
    "Gender",
    "Heart rate",
    "Systolic blood pressure",
    "Diastolic blood pressure",
    "Blood sugar",
    "CK-MB",
    "Troponin",
    "Result",
]

#Remove gender as this can mess up the graph
medical_data = medical_data.drop("Gender", axis=1)

#Set result to either 0 or 1 instead of negative or positive
label_encoder = LabelEncoder()
medical_data['Result'] = label_encoder.fit_transform(medical_data['Result'])


#Seperate result into y as this is the target
X = medical_data.drop("Result", axis=1)
X = X.values
y = medical_data["Result"]
y = y.values

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12345
)

#This trains the KNNregressor model to use the average of the 3rd nearest record in the csv file
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)

#This generates predictions based on the training data
train_preds = knn_model.predict(X_train)

#This line calculates the Root Mean Squared Error
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))

#This will display the Root Mean Squared Error for the KNNregression training data
#Root Mean Squared Error is the average amount of error in the data being used
print("KNeighborsRegressor Model:")
print("Train RMSE:", train_rmse)

#This will display the Root Mean Squared Error for the KNNregression testing data
#The 2 scores generated represent how accurate the data is compared to the actual data in the csv file
test_preds = knn_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
print("Test RMSE:", test_rmse)

#This finds the best parameters for the estimate
parameters = {"n_neighbors": range(1, 50)}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)
print("\nGrid Search Results:")
#Best parameters shows how many values to either side you would look at
#An example of this in general terms is if you were shopping for something, you would look at similar products to see which one has the best value
print("Best Parameters (KNeighbors):", gridsearch.best_params_)

#This trains a BaggingRegressor to create a seperate regression graph
best_k = gridsearch.best_params_["n_neighbors"]
bagged_knn = KNeighborsRegressor(n_neighbors=best_k)
bagging_model = BaggingRegressor(bagged_knn, n_estimators=100)
bagging_model.fit(X_train, y_train)

#This will print the Root Mean Squared Error for the Baggingregression testing data
test_preds_bagging = bagging_model.predict(X_test)
test_rmse_bagging = np.sqrt(mean_squared_error(y_test, test_preds_bagging))
print("\nBaggingRegressor Model:")
print("Bagging Test RMSE:", test_rmse_bagging)

plt.style.use('dark_background')
#This will begin the process of visualising the predicted data
plt.figure(figsize=(12, 6))

#This will plot the predicted data with KNNregression
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_preds, cmap='coolwarm')
plt.colorbar(label='Predicted likelyhood of having had a heart attack')
plt.title('KNeighborsRegressor Predictions')
plt.xlabel('Age')
plt.ylabel('Heart rate')

#This will plot the predicted data with Baggingregression
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_preds_bagging, cmap='coolwarm')
plt.colorbar(label='Predicted likelyhood of having had a heart attack')
plt.title('BaggingRegressor Predictions')
plt.xlabel('Age')
plt.ylabel('Heart rate')

#This will make the results display
plt.tight_layout()
plt.show()

