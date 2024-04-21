import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

def predict_job_title(test_data):
    #New data frame with all columns except job title as this will be predicted
    test_data_columns = ['employment_type', 'experience_level',
                   'employee_residence', 'remote_ratio', 'company_location',
                   'company_size', 'salary_in_usd']
    test_data = pd.DataFrame(test_data, columns=test_data_columns)

    #Ensure all data matches the current records
    new_data_encoded = pd.get_dummies(test_data, drop_first=True)

    #This ensures the data matches the data in the X train
    new_data_encoded = new_data_encoded.reindex(columns=X_train.columns, fill_value=0)

    #This will predict the job title of the test data entered
    predicted_job_title = dt_clf.predict(new_data_encoded)

    #Returns the predicted job title
    return predicted_job_title[0]

#Read the CSV file in the folder
salary_data = pd.read_csv("salaries.csv")

#Filter dataset to only have 2 job titles, this is for example but could easily be scaled up
salary_data_filtered = salary_data[salary_data['job_title'].isin(['Data Scientist', 'Data Engineer'])]

#salary and salary currency are discarded as salary in usd is the only relevant column
salary_data_filtered = salary_data_filtered.drop(columns=['salary', 'salary_currency'])

#This puts everything into categories so that the decision tree can function
categorical_cols = ['employment_type', 'experience_level', 'employee_residence', 'company_location', 'company_size']
salary_data_encoded = pd.get_dummies(salary_data_filtered, columns=categorical_cols, drop_first=True)

#Drop job title as this is what the decision tree is determining
X = salary_data_encoded.drop(columns=['job_title'])
y = salary_data_encoded['job_title']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=13)

#Initialize the DecisionTreeClassifier with limited depth
dt_clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2)

dt_clf.fit(X_train, y_train)

y_pred = dt_clf.predict(X_test)

#This shows how accurate the application is
print(classification_report(y_pred, y_test))

# Get the feature names
feature_names = list(X_train.columns)

#This plots the decision tree
plt.figure(figsize=(25, 25))

plot_tree(dt_clf, feature_names=feature_names, class_names=dt_clf.classes_, filled=True)

plt.show()

#This exists as a test to determine what job title someone has, in a full scale version every variable would be input
sal = input("Please enter a sample salary: ")
salint = int(sal)

#Example data for the predict job title function
test_data = {
    'employment_type': ['FT'],
    'experience_level': ['SE'],
    'employee_residence': ['US'],
    'remote_ratio': [0],
    'company_location': ['GB'],
    'company_size': ['M'],
    'salary_in_usd': [salint] 
}

#This will send back the predicted job title of the test data entry
predicted_job_title = predict_job_title(test_data)
print("Predicted job title:", predicted_job_title)