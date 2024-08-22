import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score

class ModelTrainer:
    def __init__(self, random_state=0):
        self.random_state = random_state

    def printing_KFold_scores(self, X_train_data, Y_train_data):
        kf = KFold(5, shuffle=False)
        c_param_range = [0.01, 0.1, 1, 10, 100]
        results_table = pd.DataFrame(columns=['C_parameter', 'Mean recall score'])

        for c_param in c_param_range:
            recall_accs = []
            for train_index, test_index in kf.split(Y_train_data):
                lr = LogisticRegression(C=c_param, solver='liblinear', penalty='l2')
                lr.fit(X_train_data.iloc[train_index, :], Y_train_data.iloc[train_index, :].values.ravel())
                Y_pred_undersample = lr.predict(X_train_data.iloc[test_index, :].values)
                recall_acc = recall_score(Y_train_data.iloc[test_index, :].values, Y_pred_undersample)
                recall_accs.append(recall_acc)

            mean_recall = np.mean(recall_accs)
            results_table = results_table.append({'C_parameter': c_param, 'Mean recall score': mean_recall}, ignore_index=True)

        best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
        print(f"Best C parameter from cross-validation: {best_c}")
        return best_c

    def train_logistic_regression(self, X_train, Y_train, best_c):
        model = LogisticRegression(C=best_c, solver='liblinear', penalty='l2')
        model.fit(X_train, Y_train.values.ravel())
        return model
