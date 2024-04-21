import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

Salary_data = pd.read_csv("DataScience_salaries_2024.csv")
#https://www.kaggle.com/datasets/saurabhbadole/latest-data-science-job-salaries-2024

# We'll focus on predicting salary ranges based on other features
# For simplicity, we'll only use a subset of features
features = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "remote_ratio",
    "company_size"
]

# Drop rows with missing values in any of the selected features
Salary_data = Salary_data.dropna(subset=features)

# Convert salary to integer for simplicity
Salary_data["salary_in_usd"] = Salary_data["salary_in_usd"].astype(int)

# Define a function to map salary to a range
def salary_range(salary_in_usd):
    if salary_in_usd < 110000:
        return "Low"
    elif salary_in_usd < 160000:
        return "Medium"
    else:
        return "High"

# Apply the salary range function to create the target variable
Salary_data["salary_range"] = Salary_data["salary_in_usd"].apply(salary_range)

# Assign values to the X and y variables
X = Salary_data[features].values
y = Salary_data["salary_range"].values

# One-hot encode categorical variables
X = pd.get_dummies(Salary_data[features], drop_first=True).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Standardize features by removing mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a k-nearest neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=10)

# Fit the classifier to the training data
classifier.fit(X_train, y_train)

# Predict the salary range for the test data
y_predict = classifier.predict(X_test)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)

# Create a DataFrame from the confusion matrix for better visualization
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Low', 'Actual Medium', 'Actual High'], 
                              columns=['Predicted Low', 'Predicted Medium', 'Predicted High'])

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label':'Amount of people'})
plt.xlabel('Predicted Income Groups', fontsize=14)
plt.ylabel('True Income Groups', fontsize=14)
plt.title('Confusion Matrix')
plt.show()

# Print the classification report
print(classification_report(y_test, y_predict))

