import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools

class Evaluator:
    def __init__(self):
        pass

    def evaluate_model(self, model, X_test, Y_test, title_suffix=""):
        Y_pred = model.predict(X_test.values)
        cnf_matrix = confusion_matrix(Y_test, Y_pred)
        recall_metric = cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])
        print(f"Recall metric on the {title_suffix} dataset: {recall_metric:.2f}")
        self.plot_confusion_matrix(cnf_matrix, title=f'Confusion Matrix {title_suffix}')
        print(classification_report(Y_test, Y_pred))

    def plot_confusion_matrix(self, cm, classes=[0, 1], title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f'
        thresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
