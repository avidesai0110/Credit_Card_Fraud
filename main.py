import os
from data_processing import DataProcessor
from model_training import ModelTrainer
from evaluation import Evaluator

def main():
    data_file_path = os.path.join(os.getcwd(), "data", "creditcard.csv")
    data_processor = DataProcessor(data_path=data_file_path)
    model_trainer = ModelTrainer()
    evaluator = Evaluator()

    # Preprocessing
    data_processor.preprocess_data()
    data_processor.split_data()
    data_processor.undersample_data()

    # Training and evaluating on undersampled data
    best_c_undersample = model_trainer.printing_KFold_scores(data_processor.X_train_undersample, data_processor.Y_train_undersample)
    model_undersample = model_trainer.train_logistic_regression(data_processor.X_train_undersample, data_processor.Y_train_undersample, best_c_undersample)
    evaluator.evaluate_model(model_undersample, data_processor.X_test_undersample, data_processor.Y_test_undersample, title_suffix="(Undersampled)")

    # Training and evaluating on full data
    best_c_full = model_trainer.printing_KFold_scores(data_processor.X_train, data_processor.Y_train)
    model_full = model_trainer.train_logistic_regression(data_processor.X_train, data_processor.Y_train, best_c_full)
    evaluator.evaluate_model(model_full, data_processor.X_test, data_processor.Y_test, title_suffix="(Full Dataset)")

    # Oversampling with SMOTE and evaluating
    os_features, os_labels, features_test, labels_test = data_processor.oversample_with_smote()
    best_c_smote = model_trainer.printing_KFold_scores(os_features, os_labels)
    model_smote = model_trainer.train_logistic_regression(os_features, os_labels, best_c_smote)
    evaluator.evaluate_model(model_smote, features_test, labels_test, title_suffix="(SMOTE Oversampled)")

if __name__ == "__main__":
    main()
