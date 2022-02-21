'''
    A main module.
'''
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tree_classifier import DecisionTree

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# data = datasets.load_breast_cancer()
data = datasets.load_iris()
X, y = data['data'], data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)
# print(X_train)
# print('===============================================================')
# print(y_train)
# print('===============================================================')
# print(X_test)
# print('===============================================================')
# print(y_test)

classifier = DecisionTree(max_depth=10)
classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)
y_pred = classifier.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy:", acc)
