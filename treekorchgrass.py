class Node:
    
    def __init__(self, X, y, gini):
        self.X = X
        self.y = y
        self.gini = gini
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class MyDecisionTreeClassifier:

    def __init__(self, max_depth):
        self.max_depth = max_depth

    def gini(self, groups, classes):
        '''
        A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.
        
        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem).
        '''
        all_samples = sum([len(group) for group in groups])
        gini = 0
        for group in groups:
            quantity = len(group)
            if quantity:
                ans_sq = 0
                for needed_value in classes:
                    all_results = [el[-1] for el in group if el[-1] == needed_value]
                    probability = len(all_results) / quantity
                    ans_sq += probability**2
                gini += (1.0 - ans_sq) * (quantity / all_samples)
        return gini
 

    def split_data(self, X, y): #-> tuple[int, int]:
        # test all the possible splits in O(N^2)
        # return index and threshold value
        
        # class_values = list(set(row[-1] for row in dataset))
        # b_index, b_value, b_score, b_groups = 999, 999, 999, None
        # for index in range(len(dataset[0])-1):
        #     for row in dataset:
        #         groups = test_split(index, row[index], dataset)
        #         gini = gini_index(groups, class_values)
        #         if gini < b_score:
        #             b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        # return {'index':b_index, 'value':b_value, 'groups':b_groups}
        pass

    def build_tree(self, X, y, depth = 0):
        # create a root node
        # recursively split until max depth is not exeeced
        
        pass
    
    def fit(self, X, y):
        # basically wrapper for build tree
        
        pass
    
    def predict(self, X_test):
        # traverse the tree while there is left node
        # and return the predicted class for it, 
        # note that X_test can be not only one example
        
        pass


# tree = MyDecisionTreeClassifier(3)
# print(tree.gini([[[0, 1], [0, 0]], [[1, 1], [1, 1]]], [0, 1]))
