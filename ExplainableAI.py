import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

def predict_job_title(new_data_point):
    # Ensure the new data point has all the necessary columns except salary_currency
    all_columns = ['employment_type', 'experience_level',
                   'employee_residence', 'remote_ratio', 'company_location',
                   'company_size', 'salary_in_usd']
    new_data_point = pd.DataFrame(new_data_point, columns=all_columns)

    # Perform one-hot encoding for the categorical columns
    new_data_encoded = pd.get_dummies(new_data_point, drop_first=True)

    # Reindex the new_data_encoded to match the columns in X_train
    new_data_encoded = new_data_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Use the trained model to predict the job title
    predicted_job_title = dt_clf.predict(new_data_encoded)

    return predicted_job_title[0]

# Read the CSV file
salary_data = pd.read_csv("salaries.csv")

# Filter the dataset to include only "Data Scientist" and "Data Analyst" job titles
salary_data_filtered = salary_data[salary_data['job_title'].isin(['Data Scientist', 'Data Engineer'])]

# Drop 'salary' and 'salary_currency' columns
salary_data_filtered = salary_data_filtered.drop(columns=['salary', 'salary_currency'])

# Perform one-hot encoding for categorical columns
categorical_cols = ['employment_type', 'experience_level', 'employee_residence', 'company_location', 'company_size']
salary_data_encoded = pd.get_dummies(salary_data_filtered, columns=categorical_cols, drop_first=True)

# Drop 'job_title' column since it's the target variable
X = salary_data_encoded.drop(columns=['job_title'])
y = salary_data_encoded['job_title']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=13)

# Initialize the DecisionTreeClassifier with limited depth
dt_clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2)

# Train the classifier
dt_clf.fit(X_train, y_train)

# Predict on the test data
y_pred = dt_clf.predict(X_test)

# Evaluate the model
print(classification_report(y_pred, y_test))

# Get the feature names
feature_names = list(X_train.columns)

# Plot the decision tree with adjusted plot size
plt.figure(figsize=(25, 25))

plot_tree(dt_clf, feature_names=feature_names, class_names=dt_clf.classes_, filled=True)
plt.show()

sal = input("Please enter a sample salary: ")
salint = int(sal)

# Example usage of the predict_job_title function
new_data_point = {
    'employment_type': ['FT'],
    'experience_level': ['SE'],
    'employee_residence': ['US'],
    'remote_ratio': [0],
    'company_location': ['GB'],
    'company_size': ['M'],
    'salary_in_usd': [salint]  # Example salary in USD
}

predicted_job_title = predict_job_title(new_data_point)
print("Predicted job title:", predicted_job_title)