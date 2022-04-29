import copy
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import random


class DTC(DecisionTreeClassifier):

    def __init__(self, alpha, iterations, max_depth=None, min_samples_split=2):
        self.alpha = alpha
        self.iterations = iterations
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)

    def predict(self, X, check_input=True):
        """
        Overriding Sklearn predict method.
        :param X:
        :param check_input:
        :return:
        """
        ground_truth_thresh = copy.copy(self.tree_.threshold)
        preds = []
        results = pd.DataFrame()

        for num in range(self.iterations):
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

            results['round' + str(num)] = preds
            preds = []

        return self.calculate_avg(results)

    def calculate_avg(self, results):
        """
        For n iterations, get the most common prediction for a sample.
        :param results:
        :return:
        """

        most_common_values = results.mode(axis=1)
        return most_common_values.iloc[:, 0]

    def manipulate_thresholds(self, X, alpha):
        """
        Change node threshold with a certain proba (alpha).
        :param X:
        :param alpha:
        :return:
        """

        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        n_nodes = self.tree_.node_count
        feature = self.tree_.feature
        threshold = self.tree_.threshold

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]
        while len(stack) > 0:
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            is_split_node = children_left[node_id] != children_right[node_id]
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))

                feature_col = feature[node_id]
                random_val = random.uniform(0, 1)

                if random_val <= alpha:
                    if threshold[node_id] <= X[:, feature_col]:
                        threshold[node_id] = X[:, feature_col] + 1
                    else:
                        threshold[node_id] = X[:, feature_col] - 1
            else:
                is_leaves[node_id] = True


