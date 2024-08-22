import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path, test_size=0.3, random_state=0):
        self.data = pd.read_csv(data_path)
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None
        self.X_train_undersample, self.X_test_undersample = None, None
        self.Y_train_undersample, self.Y_test_undersample = None, None

    def preprocess_data(self):
        self.data['normAmount'] = StandardScaler().fit_transform(self.data['Amount'].values.reshape(-1, 1))
        self.data = self.data.drop(['Time', 'Amount'], axis=1)
        print("Data preprocessing completed.")

    def undersample_data(self):
        X = self.data.iloc[:, self.data.columns != 'Class']
        Y = self.data.iloc[:, self.data.columns == 'Class']

        fraud_indices = np.array(self.data[self.data.Class == 1].index)
        normal_indices = self.data[self.data.Class == 0].index

        random_normal_indices = np.random.choice(normal_indices, len(fraud_indices), replace=False)
        under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

        under_sample_data = self.data.iloc[under_sample_indices, :]

        self.X_train_undersample, self.X_test_undersample, self.Y_train_undersample, self.Y_test_undersample = train_test_split(
            under_sample_data.iloc[:, under_sample_data.columns != 'Class'],
            under_sample_data.iloc[:, under_sample_data.columns == 'Class'],
            test_size=self.test_size,
            random_state=self.random_state
        )
        print("Undersampling completed.")

    def split_data(self):
        X = self.data.iloc[:, self.data.columns != 'Class']
        Y = self.data.iloc[:, self.data.columns == 'Class']

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)
        print(f"Data split into training and test sets with test size {self.test_size}.")

    def oversample_with_smote(self):
        from imblearn.over_sampling import SMOTE
        features_columns = self.data.columns.delete(len(self.data.columns) - 1)
        features = self.data[features_columns]
        labels = self.data['Class']

        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=self.test_size, random_state=self.random_state)

        smote = SMOTE(random_state=self.random_state)
        os_features, os_labels = smote.fit_resample(features_train, labels_train)
        print(f"Oversampling completed. Number of samples in the minority class after oversampling: {len(os_labels[os_labels == 1])}")

        return os_features, os_labels, features_test, labels_test
