'''
    A module that implements TreeClassifier class.
'''
from cgi import print_directory
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


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
    def __init__(self, max_depth=15):
        self.max_depth = max_depth
        self.root = None

    def _is_finished(self, depth):
        if (depth >= self.max_depth
            or self.n_class_labels == 1
                or self.samples_count < 2):
            return True
        return False

    def _build_tree(self, dataset, target, depth=0):
        self.samples_count, self.features_count = dataset.shape
        self.n_class_labels = len(set(target))

        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(target))
            return Node(value=most_common_Label)

        rnd_feats = np.random.choice(
            self.features_count, self.features_count, replace=False)
        best_feat, best_thresh = self._best_split(dataset, target, rnd_feats)
        left_part, right_part = self.__split_data(
            dataset[:, best_feat], best_thresh)
        left_child = self._build_tree(
            dataset[left_part, :], target[left_part], depth + 1)
        right_child = self._build_tree(
            dataset[right_part, :], target[right_part], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def fit(self, dataset, target):
        self.root = self._build_tree(dataset, target)

    def gini_impurity(self, target):
        proportions = np.bincount(target) / len(target)
        # gini = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        gini = 1 - np.sum([p*p for p in proportions])
        return gini

    def __split_data(self, dataset, thresh):
        left_part = np.argwhere(dataset <= thresh).flatten()
        right_part = np.argwhere(dataset > thresh).flatten()
        return left_part, right_part

    def __information_gain(self, dataset, target, thresh):
        parent_impurity = self.gini_impurity(target)
        left_part, right_part = self.__split_data(dataset, thresh)
        cnt, cnt_left, cnt_right = len(target), \
            len(left_part), len(right_part)

        if cnt_left == 0 or cnt_right == 0:
            return 0

        child_impurity = (cnt_left / cnt) * self.gini_impurity(target[left_part]) + \
            (cnt_right / cnt) * self.gini_impurity(target[right_part])
        return parent_impurity - child_impurity

    def _best_split(self, dataset, target, features):
        split = {'score': - 1, 'feat': None, 'thresh': None}
        for feat in features:
            X_feat = dataset[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self.__information_gain(X_feat, target, thresh)
                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']

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
            For each row in dataset dataset (who the fuck names dataset dataset???), funtions
            traverses our tree to get to the leaf node. Then, the array of the
            values of each leaf will be returned
        Args:
            dataset (iterable of iterables): a dataset. May be an array of arrays.
        Returns:
            np.array: _description_
        """
        predictions = [self.traverse(dataset, self.root)
                       for dataset in dataset]
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
