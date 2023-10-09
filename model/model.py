import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

# ================0. DATA================= #
# Import Data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Check unique value
for i in train_data.columns:
    no_of_unique = len(train_data[i].unique())
    print(f"{i} => {no_of_unique}")
"""
From unique value, we see that chol and thalach have many unique values.
"""

# ================1. EDA================= #
# Basic Statistics
print(train_data.describe())
"""
From basic statistics, we see that the minimum age is 29 and maximum age is 77.
"""
# Density of age, chol and thalach
sns.histplot(train_data['age'])
plt.show()
sns.histplot(train_data['chol'])
plt.show()
sns.histplot(train_data['thalach'])
plt.show()

# Correlation Matrix
corr = train_data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.show()

# ================2. FEATURE SCALING================= #
scaler = MinMaxScaler()

X = train_data.drop(['target'], axis=1)
Y = train_data['target']
X = scaler.fit_transform(X)

X_TEST = test_data.drop(['target'], axis=1)
Y_TEST = test_data['target']
X_TEST = scaler.fit_transform(X_TEST)

# ================3. MODEL================= #
# SVM
svm_grid = GridSearchCV(svm.SVC(), {'C': [1, 10, 25], 'kernel': ['linear', 'poly', 'rbf'], 'degree': [1, 2, 3],
                                    'gamma': ['scale', 'auto']}, cv=5, return_train_score=False)
svm_grid.fit(X, Y)
# print(svm_grid.best_params_)
svm = svm.SVC(C=25, degree=3, gamma='scale', kernel='poly')
svm.fit(X, Y)
y_pred = svm.predict(X_TEST)
svm_score = accuracy_score(y_pred, Y_TEST)

# Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
rfc = RandomForestClassifier()
rf_Grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, verbose=2, n_jobs=4)
rf_Grid.fit(X, Y)
rf = RandomForestClassifier(min_samples_leaf=1, min_samples_split=2, n_estimators=100)
rf.fit(X, Y)
y_pred = rf.predict(X_TEST)
rf_score = accuracy_score(Y_TEST, y_pred)

# Logistic Regression
lgr = LogisticRegression()
param_grid = {
    'penalty': ['l1', 'l2'],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky'],
    'max_iter': [1000, 1500, 2000],
    'multi_class': ['auto', 'ovr', 'multinomial'],
    'class_weight': ['dict', 'balanced']
}
lgr_grid = GridSearchCV(estimator=lgr, param_grid=param_grid, cv=3, verbose=2, n_jobs=4)
lgr_grid.fit(X, Y)
log = LogisticRegression(class_weight='balanced', max_iter=1000, multi_class='auto', penalty='l2', solver='liblinear')
log.fit(X, Y)
y_pred = log.predict(X_TEST)
lgr_score = accuracy_score(Y_TEST, y_pred)

# Conclusion
print(
    f"Accuracy of SVM = {svm_score}\nAccuracy of Random Forest = {rf_score}\nAccuracy of Logistic Regression = {lgr_score}")

# ================4. EXPORT MODEL================= #
joblib.dump(rf, "my_ml_model_eiei.joblib")
