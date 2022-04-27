import copy

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import random


class DTC(DecisionTreeClassifier):

    def __init__(self, alpha, max_depth=None, min_samples_split=2):
        self.alpha = alpha
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)

    def predict(self, X, check_input=True):
        ground_truth_thresh = copy.copy(self.tree_.threshold)
        preds = []
        for index, row in X.iterrows():
            row = row.array.reshape(1, -1)
            row = self._validate_X_predict(row, check_input)
            self.manipulate_thresholds(row, self.alpha)
            proba = self.tree_.predict(row)
            n_samples = row.shape[0]

            # Classification
            if self.n_outputs_ == 1:
                pred_val = self.classes_.take(np.argmax(proba, axis=1), axis=0)
                preds.append(pred_val[0])

                for i in range(len(self.tree_.threshold)):
                    self.tree_.threshold[i] = ground_truth_thresh[i]

            else:
                class_type = self.classes_[0].dtype
                predictions = np.zeros((n_samples, self.n_outputs_), dtype=class_type)
                for k in range(self.n_outputs_):
                    predictions[:, k] = self.classes_[k].take(
                        np.argmax(proba[:, k], axis=1), axis=0
                    )

                return predictions

        return preds

    def manipulate_thresholds(self, X, alpha):
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        n_nodes = self.tree_.node_count
        feature = self.tree_.feature
        threshold = self.tree_.threshold

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))

                feature_col = feature[node_id]
                random_val = random.uniform(0, 1)

                if random_val <= alpha:
                    if threshold[node_id] <= X[:, feature_col]:
                        try:
                            threshold[node_id] = X[:, feature_col] + 1
                        except:
                            print(X[:, feature_col])
                            print(X[:, feature_col] + 1)
                    else:
                        threshold[node_id] = X[:, feature_col] - 1
            else:
                is_leaves[node_id] = True


