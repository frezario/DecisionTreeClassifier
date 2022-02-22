'''
    A main module with examples.
'''
from cgi import test
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tree_classifier import DecisionTree

def precision(result, true_result):
    precision = np.sum(true_result == result) / len(true_result)
    return precision

clf = DecisionTree(max_depth=15)

data = datasets.load_iris()
dataset, target = data['data'], data['target']
for test_size in range(1, 10, 1):
    dataset_train, dataset_test, target_train, target_test = train_test_split(
        dataset, target, test_size= test_size * 0.1, random_state=1
    )
    clf.fit(dataset_train, target_train)
    result = clf.predict(dataset_test)
    print(f"Test size: {100 - test_size * 10} % of dataset. Tree accuracy:", precision(result, target_test))
