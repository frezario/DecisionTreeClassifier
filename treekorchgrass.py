from numpy import inf


class Node:

    def __init__(self, X, y, gini):
        self.X = X
        self.y = y
        self.gini = gini
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

    def gini(self, groups, classes):
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
        all_samples = len(groups) * len(groups[0])
        for group in groups:
            if group:
                quantity = len(group)
                ans_sq = []
                for needed_value in classes:
                    all_results = [el[-1] for el in group if el[-1] == needed_value]
                    probability = len(all_results) / quantity
                    ans_sq.append(probability**2)
                gini_impurity.append((1.0 - sum(ans_sq)) * (quantity / all_samples))
        return sum(gini_impurity)

class MyDecisionTreeClassifier:

    def __init__(self, max_depth):
        self.max_depth = max_depth

    # @staticmethod
    def gini(self, groups, classes):
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
        all_samples = len(groups) * len(groups[0])
        for group in groups:
            if group:
                quantity = len(group)
                ans_sq = []
                for needed_value in classes:
                    all_results = [el[-1] for el in group if el[-1] == needed_value]
                    probability = len(all_results) / quantity
                    ans_sq.append(probability**2)
                gini_impurity.append((1.0 - sum(ans_sq)) * (quantity / all_samples))
        return sum(gini_impurity)


    def split_data(self, X, y): #-> tuple[int, int]: (data, target)
        """
        tests all the possible splits in O(N^2)
        returns index and threshold value
        Seperates a table into two lists: satisfy the condition,
        or doesn't.
        Args:
            X (_type_): group name
            y (_type_): value
        """
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            # We could actually split the node according to each feature/threshold pair
            # and count the resulting population for each class in the children, but
            # instead we compute them in an iterative fashion, making this for loop
            # linear rather than quadratic.
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )

                # The Gini impurity of a split is the weighted average of the Gini
                # impurity of the children.
                gini = (i * gini_left + (m - i) * gini_right) / m

                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_idx, best_thr


    def build_tree(self, X, y, depth = 0):
        # create a root node
        # recursively split until max depth is not exeeced
        # create a root node, continue recursively and stop if depth is exceeded

        pass
    
    def fit(self, X, y):
        # basically wrapper for build tree
        
        pass

    def predict(self, X_test):
        # traverse the tree while there is left node
        # and return the predicted class for it, 
        # note that X_test can be not only one example

        pass


    def test_split(ind, pivot, dataset):
        left = [row for row in dataset if row[ind] < pivot]
        right = [row for row in dataset if row[ind] >= pivot]
        return left, right


    # Select the best split point for a dataset
    def split_data(dataset):
        class_values = list(set(row[-1] for row in dataset))
        print(class_values)
        best_index, best_gini, best_score, best_groups = inf, inf, inf, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = test_split(index, row[index], dataset)
                gini = gini(groups, class_values)
                if gini < best_score:
                    best_index, best_gini, best_score, best_groups = index, row[index], gini, groups
        # return {'index':best_index, 'value':best_gini, 'groups':best_groups}
        # rewrite the return into oop 

# tree = MyDecisionTreeClassifier(3)
# print(tree.gini([[[0, 1], [0, 0]], [[1, 1], [1, 1]]], [0, 1]))

from filereader import reading_file
dataset = reading_file("SecondTask/iris.csv")

dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
split = get_split(dataset)
print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))
