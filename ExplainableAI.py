import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

salary_data = pd.read_csv("salaries.csv")

salary_data.hist(bins=25,figsize=(10,10))
# display histogram
import matplotlib.pyplot as plt

plt.show()


X = salary_data.drop(columns='experience_level')
y = salary_data['experience_level']


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,
                                                    stratify =y,
                                                    random_state = 13)

from sklearn.tree import DecisionTreeClassifier, plot_tree
dt_clf = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 2)
dt_clf.fit(X_train, y_train)


# Predict on the test data and evaluate the model
y_pred = dt_clf.predict(X_test)


print(classification_report(y_pred, y_test))

# Get the class names
class_names = ['Money earned', 'No money earned']


# Get the feature names
feature_names = list(X_train.columns)


fig = plt.figure(figsize=(25,20))
_ = plot_tree(dt_clf,
                feature_names = feature_names,
                class_names = class_names,
                filled=True)
plt.show()

#Make the decision tree predict which job title someone will have
#drop all job title records apart from 2 