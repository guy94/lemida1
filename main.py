from sklearn.model_selection import GridSearchCV
import numpy as np
import read_csv as reader
from data_preparation import DataPreparation
from models import DTC
from sklearn.metrics import f1_score, balanced_accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def iterate_files():
    """
    for each csv file - read, prepare, split for train and test sets.
    :return:
    """

    file_names = reader.read_file_names('data')
    total_data = {}
    for name in file_names:
        dp = DataPreparation()
        dp.df = reader.read_file(name)
        prepare_data(dp, name)
        total_data[name] = dp

    return total_data


def prepare_data(dp, file_name):
    """
    calls all relevant functions from data_preparation class
    :param dp:
    :return:
    """

    dp.partition_data_sets()
    dp.fill_na()
    # correlation(dp, file_name)
    for i in range(len(dp.all_data_frames)):
        dp.discretization(dp.all_data_frames[i])


def correlation(dp, file_name):
    """
    plot correlation heatmap of the features of each data set.
    :param dp:
    :return:
    """
    plt.figure(figsize=(14, 10))
    plt.title(file_name.split('/')[1].split('.')[0])
    cor = dp.df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()


def print_best_params(best_params, name):
    """
    print the selected values of each hyper-param
    :param best_params:
    :return:
    """

    print(f"\n\nDecision Tree Best Params For Data Set: {name.split('/')[1]}")
    for key, val in best_params.items():
        print(f'{key}: {val}')


def run_decision_tree_model(model, grid_search=False, is_modified=False):
    """
    run DecisionTreeClassifier model,
    consider whether to apply grid search with cross validation for hyper parameters tuning.
    evaluate the model using F1 Score metric.
    :return:
    """

    total_data = iterate_files()
    class_model = model
    f1_scores = []
    balanced_accuracy = []

    for name, prepared_data in total_data.items():
        #     if grid_search:
        #         params = {'max_depth': [5, 10, 20, 50], 'min_samples_split': [2, 3, 5]}
        #         dtc_gs = GridSearchCV(class_model, params, cv=5).fit(prepared_data.x_train,
        #                                                                    np.ravel(prepared_data.y_train))
        #         best_params = dtc_gs.best_params_
        #
        #         if is_modified:
        #             class_model = DTC(class_model.alpha, class_model.iterations, max_depth=best_params['max_depth'],
        #                           min_samples_split=best_params['min_samples_split'])
        #         else:
        #             class_model = DecisionTreeClassifier()
        #
        # class_model.fit(prepared_data.x_train.values, np.ravel(prepared_data.y_train))
        # y_prediction = class_model.predict(prepared_data.x_test)
        # f1_scores.append([name.split('/')[1], evaluate_f1(y_prediction, prepared_data.y_test)])
        # balanced_accuracy.append([name.split('/')[1], evaluate_balanced_accuracy(y_prediction, prepared_data.y_test)])

        for i in range(2):
            kf = StratifiedKFold(n_splits=5)
            for train_index, test_index in kf.split(prepared_data.x, prepared_data.y):
                X_train, X_test, y_train, y_test = prepared_data.x.iloc[train_index], prepared_data.x.iloc[test_index], \
                                                   prepared_data.y.iloc[train_index], prepared_data.y.iloc[test_index]

                class_model.fit(X_train.values, np.ravel(y_train))
                y_prediction = class_model.predict(X_test)
                f1_scores.append(evaluate_f1(y_prediction, y_test))
                balanced_accuracy.append(evaluate_balanced_accuracy(y_prediction, y_test))
    return round(np.average(f1_scores), 3), round(np.average(balanced_accuracy), 3)


def evaluate_f1(y_test, y_pred):
    """
    apply F1 Score metric, calculating the weighted average of all classes classifications.
    considers both Recall and Precision metrics.
    :param y_test:
    :param y_pred:
    :return:
    """
    score = f1_score(y_test, y_pred, average='weighted')
    return score


def evaluate_balanced_accuracy(y_test, y_pred):
    """
    apply balance accuracy Score metric, calculating the weighted average of all classes classifications.
    :param y_test:
    :param y_pred:
    :return:
    """
    score = balanced_accuracy_score(y_test, y_pred)
    return score


if __name__ == '__main__':

    alphas = [0, 0.1, 0.3, 0.5, 1]
    rounds = [30, 65, 100]
    decision_tree = DecisionTreeClassifier()
    modified_tree_f1_scores = []
    modified_tree_balanced_accuracy_scores = []

    f1_scores, balanced_accuracy_scores = run_decision_tree_model(decision_tree, grid_search=True)
    print(f'\nRegular Decision Tree F1 Score: {f1_scores}')
    print(f'Regular Decision Tree Balanced Accuracy Score: {balanced_accuracy_scores}')

    for alpha in alphas:
        for iterations in rounds:
            our_decision_tree = DTC(alpha=alpha, iterations=iterations)

            f1_scores, balanced_accuracy_scores = run_decision_tree_model(our_decision_tree, grid_search=True,
                                                                          is_modified=True)
            print(f'\nExecution Parameters: [alpha={alpha}, iterations={iterations}]')
            print(f'\tModified Decision Tree F1 Score: {f1_scores}')
            print(f'\tModified Decision Tree Balanced Accuracy Score: {balanced_accuracy_scores}')

            modified_tree_f1_scores.append([alpha, iterations, f1_scores])
            modified_tree_balanced_accuracy_scores.append([alpha, iterations, balanced_accuracy_scores])

    f1_scores_df = pd.DataFrame(modified_tree_f1_scores, columns=['alpha', 'iterations', 'f1_score'])
    balanced_accuracy_df = pd.DataFrame(modified_tree_balanced_accuracy_scores, columns=['alpha', 'iterations',
                                                                                         'balanced_accuracy_score'])
    f1_scores_df.to_csv('f1_scores.csv')
    balanced_accuracy_df.to_csv('balanced_accuracy_scores.csv')
