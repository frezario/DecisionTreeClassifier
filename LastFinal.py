'''
    A module that implements TreeClassifier class.
'''
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
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

    def is_complete(self, depth):
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

    def build_tree(self, dataset, target, depth=0):
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

        if self.is_complete(depth):
            most_common_class = np.argmax(np.bincount(target))
            return Node(value=most_common_class)
        
        # picking random order of the features
        rand_features = np.random.choice(
            self.features_count, self.features_count, replace=False)
        # creating a split on this node
        best_feature, best_threshold = self.best_split(dataset, target, rand_features)
        # splitting samples into right and left parts
        left_part, right_part = self.split(
            dataset[:, best_feature], best_threshold)


        left_part, right_part, best_feature, best_threshold = self.split_data(dataset, target)
        # getting indexes out of samples
        left_part = [ind for ind in range(len(dataset)) if dataset[ind][best_feature] < best_threshold]
        right_part = [ind for ind in range(len(dataset)) if dataset[ind][best_feature] >= best_threshold]

        # building left subtree
        left_child = self.build_tree(
            dataset[left_part, :], target[left_part], depth + 1)
        # building right subtree
        right_child = self.build_tree(
            dataset[right_part, :], target[right_part], depth + 1)
        return Node(best_feature, best_threshold, left_child, right_child)

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
        '''
            A public wrapper for build_tree() method.
        '''
        self.root = self.build_tree(dataset, target)

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
            For each row in dataset, funtion traverses our tree to get
            to the leaf node. Then, the array of the
            values of each leaf will be returned.
        Args:
            dataset (iterable of iterables): a dataset. May be an array of arrays.
        Returns:
            np.array: a prediction for this dataset
        """
        predictions = [self.traverse(dataset, self.root)
                       for dataset in dataset]
        return np.array(predictions)

    @staticmethod
    def print_tree(node: Node, depth=0):
        '''
            Prints a tree recursively.
        '''
        if node.is_leaf():
            print('\t' * depth +
                  f'This node is a leaf. Value : {node.value}.')
            return
        print('\t' * depth +
              f'This node is internal. Feature : {node.feature} Treshhold : {node.threshold}.')
        DecisionTree.print_tree(node.left, depth+1)
        DecisionTree.print_tree(node.right, depth+1)