'''
    A module that implements TreeClassifier class.
'''
import numpy as np
from sklearn import datasets


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
        A decision tree classifier class.
    """

    def __init__(self, max_depth=15):
        """
            A constructor.
        Args:
            max_depth (int, optional): a maximum height to continue build a tree. Defaults to 15.
        """
        self.max_depth = max_depth
        self.root = None
        self.samples_count = 0
        self.features_count = 0
        self.class_count = 0

    def __is_finished(self, depth):
        '''
            Returns True if no division can be done and False otherwise.
            (if there is one class only or if max depth exceeded or if
            threre is less or equal than one samples.)
        '''
        if (self.class_count == 1
            or depth >= self.max_depth
                or self.samples_count <= 1):
            return True
        return False

    def __build_tree(self, dataset, target, depth=0):
        """
            Builds a best possible tree recursively.
            If tree is finished, returns a leaf (a Node with a value).
        Args:
            dataset (_type_): a dataset
            target (_type_): classses
            depth (int, optional): a depth of recursion. Defaults to 0.
        """    
        self.samples_count, self.features_count = dataset.shape
        self.class_count = len(set(target))

        if self.__is_finished(depth):
            most_common_class = np.argmax(np.bincount(target))
            return Node(value=most_common_class)

        rand_featuress = np.random.choice(
            self.features_count, self.features_count, replace=False)
        best_feature, best_threshold = self.__best_split(dataset, target, rand_featuress)
        left_part, right_part = self.__split_data(
            dataset[:, best_feature], best_threshold)
        left_child = self.__build_tree(
            dataset[left_part, :], target[left_part], depth + 1)
        right_child = self.__build_tree(
            dataset[right_part, :], target[right_part], depth + 1)
        return Node(best_feature, best_threshold, left_child, right_child)

    def fit(self, dataset, target):
        self.root = self.__build_tree(dataset, target)

    def gini_impurity(self, target):
        '''
            Calculates the gini impurity for a certain target.
        '''
        proportions = np.bincount(target) / len(target)
        gini = 1 - np.sum([p*p for p in proportions])
        return gini

    def __split_data(self, dataset, threshold):
        '''
            Splits samples via threshold.
        '''
        left_part = np.argwhere(dataset <= threshold).flatten()
        right_part = np.argwhere(dataset > threshold).flatten()
        return left_part, right_part

    def __information_gain(self, dataset, target, threshold):
        '''
            Measures information gain (the measure of how much
            impurity we've lost).
        '''
        parent_impurity = self.gini_impurity(target)
        left_part, right_part = self.__split_data(dataset, threshold)
        cnt, cnt_left, cnt_right = len(target), len(left_part), len(right_part)
        # The worst case
        if cnt_left == 0 or cnt_right == 0:
            return 0
        # Gain = 1 - gini(split)
        child_impurity = (cnt_left / cnt) * self.gini_impurity(target[left_part]) + \
            (cnt_right / cnt) * self.gini_impurity(target[right_part])
        return parent_impurity - child_impurity

    def __best_split(self, dataset, target, features):
        # split list is in format [score, feature, treshhold]
        split = [-1, None, None]
        for feat in features:
            feature = dataset[:, feat]
            thresholds = np.unique(feature)
            for threshold in thresholds:
                score = self.__information_gain(feature, target, threshold)
                # if we found best case, we will accept it.
                if score > split[0]:
                    split[0] = score
                    split[1] = feat
                    split[2] = threshold
        return split[1], split[2]

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
            For each row in dataset, funtiontraverses our tree to get 
            to the leaf node. Then, the array of the
            values of each leaf will be returned.
        Args:
            dataset (iterable of iterables): a dataset. May be an array of arrays.
        Returns:
            np.array: _description_
        """
        predictions = [self.traverse(dataset, self.root)
                       for dataset in dataset]
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
