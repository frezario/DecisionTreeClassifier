'''
    A module that implements TreeClassifier class.
'''
from turtle import left
import numpy as np
from sklearn import datasets
from numpy import inf


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    """
        A decisiomn tree classifier.
    """
    def __init__(self, max_depth=15):
        self.max_depth = max_depth
        self.root = None


    def __is_finished(self, depth):
        if (depth >= self.max_depth
            or self.n_class_labels == 1
                or self.samples_count < 2):
            return True
        return False


    def __build_tree(self, dataset, target, depth=0):
        self.samples_count, self.features_count = dataset.shape
        self.class_count = len(set(target))

        if self.is_complete(depth):
            most_common_class = np.argmax(np.bincount(target))
            return Node(value=most_common_class)
        
        left_part, right_part, best_feat, best_threshold = self.split_data(dataset, target)
        
        # getting indexes out of samples
        left_part = [ind for ind in range(len(dataset)) if dataset[ind][best_feat] < best_feat]
        right_part = [ind for ind in range(len(dataset)) if dataset[ind][best_feat] >= best_feat]
        
        # building left subtree
        left_child = self.build_tree(
            dataset[left_part, :], target[left_part], depth + 1)
        # building right subtree
        right_child = self.build_tree(
            dataset[right_part, :], target[right_part], depth + 1)
        return Node(best_feat, best_threshold, left_child, right_child)

def gini(self, groups, target):
        '''
        A Gini score gives an idea of how good a split is by how mixed
        the classes are in the two groups created by the split.
        
        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem).

        Args:
            groups (list): a column in a dataset
            classes (list): a list of possible outcomes (flower types)
        '''
        gini_impurity = []
        all_samples = sum([len(groups[i]) for i in range(len(groups))])
        for column in groups:
            if column:
                quantity = len(column)
                ans_sq = []
                for needed_value in target:
                    print(f"Group: {column}")
                    all_results = [el[-1] for el in column if el[-1] == needed_value]
                    probability = len(all_results) / quantity
                    ans_sq.append(probability**2)
                gini_impurity.append((1.0 - sum(ans_sq)) * (quantity / all_samples))
        return sum(gini_impurity)


    @staticmethod
    def test_split(ind, pivot, dataset):
        """
        Seperates a table into two lists: elements from
        first one satisfy the condition, from second one don't.
        """
        left = [row for row in dataset if row[ind] < pivot]
        right = [row for row in dataset if row[ind] >= pivot]
        return left, right

    # Select the best split point for a dataset
    def split_data(self, dataset, target):
        """
        tests all the possible splits in O(N^2)
        returns index and threshold value
        Args:
            dataset (_type_): group name
            y (_type_): targets, for example [0, 1, 2]
        """
        # if not target:
        #     target = list(set(row[-1] for row in dataset)) # change it - repeated action
        # print(class_values)
        best_index, pivot_value, best_gini, best_groups = None, inf, inf, None # self.gini(dataset, target)
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini(groups, target)
                if gini < best_gini:
                    # best_index, pivot_value, best_gini, best_groups = index, row[index], gini, groups
                    best_index, pivot_value, best_gini = index, row[index], gini

        return best_groups[0], best_groups[1], best_index, pivot_value


    def fit(self, dataset, target):
        self.root = self.__build_tree(dataset, target)


    def traverse(self, piece_of_data, node: Node):
        """
            Traverses the tree right to the leaf recursively.
        Args:
            piece_of_data (_type_): an array of features.
            node (Node): a node to start a traversal.
        Returns:
            an affiliation class: a result of classifying an object.
        """
        # If node is a leaf, we will return it's value and finish.
        if node.is_leaf():
            return node.value
        # Otherwise, we will go to subtrees
        if piece_of_data[node.feature] <= node.threshold:
            # To the left, if our feature is less then a treshold
            return self.traverse(piece_of_data, node.left)
        # To the right otherwise
        return self.traverse(piece_of_data, node.right)


    def predict(self, dataset):
        """
            For each row in dataset, funtions
            traverses our tree to get to the leaf node. Then, the array of the
            values of each leaf will be returned
        Args:
            dataset (iterable of iterables): a dataset. May be an array of arrays.
        Returns:
            np.array: _description_
        """
        predictions = [self.traverse(dataset, self.root)
                       for data in dataset]
        # return predictions
        return np.array(predictions)

    @staticmethod
    def print_tree(node: Node, depth=0):
        if node.is_leaf():
            print('\t' * depth +
                  f'This node is a leaf. It\'s value is {node.value}.')
            return
        print('\t' * depth +
              f'This node is not terminal one. It\'s treshhold is {node.threshold}.')
        DecisionTree.print_tree(node.left, depth+1)
        DecisionTree.print_tree(node.right, depth+1)


clf = DecisionTree(max_depth=10)

data = datasets.load_iris()
dataset, target = data['data'], data['target']
clf.fit(dataset, target)
# clf.print_tree(clf.root)
print(clf.predict(dataset))
