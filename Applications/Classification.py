import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#This application can determine which catagory of earning someone could fall into depending on other factors

#Load the data
Salary_data = pd.read_csv("DataScience_salaries_2024.csv")

#Use a number of features to predict salary group
features = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "remote_ratio",
    "company_size",
]

#This gets rid of rows with missing pieces of data
Salary_data = Salary_data.dropna(subset=features)

#Convert the salary_in_usd figure to integer for comparison
Salary_data["salary_in_usd"] = Salary_data["salary_in_usd"].astype(int)

#Define custom salary range function for 2 categories
threshold = Salary_data['salary_in_usd'].quantile(0.5)  #2 quartiles, low being bottom 50% and high being top 50%

def salary_range_custom(salary_in_usd):
    if salary_in_usd <= threshold:
        return "Low"
    else:
        return "High"

#This creates a new salary range column
Salary_data["salary_range_custom"] = Salary_data["salary_in_usd"].apply(salary_range_custom)

#This assigns the new salary range column to y
y = Salary_data["salary_range_custom"]

#This uses one hot encoding to categorize values, this format is more useful for machine learning
X = Salary_data[features]
X = pd.get_dummies(X, drop_first=True)

#This splits the data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#This scales the values to uniform variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#This improves accuracy using multiple decision trees
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

#This makes the classifier use the training data
classifier.fit(X_train_scaled, y_train)

#This makes the application predict the salary range for the test data
y_pred = classifier.predict(X_test_scaled)

#This calculates the confusion matrix for the data
conf_matrix = confusion_matrix(y_test, y_pred)

#This will be used to create the confusion matrix
sorted_labels = sorted(y.unique(), reverse=True)

#This creates a dataframe to use with the confusion matrix
conf_matrix_df = pd.DataFrame(conf_matrix, index=sorted_labels, columns=sorted_labels)

#This plots the confusion matrix as a heat map
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Amount of people'})
plt.xlabel('Predicted Income Groups', fontsize=14)
plt.ylabel('True Income Groups', fontsize=14)
plt.title('Confusion Matrix')
plt.show()

#This prints the classification report, showing the F1 score
print(classification_report(y_test, y_pred))