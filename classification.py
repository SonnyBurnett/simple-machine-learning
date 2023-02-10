import data_prep
import ml_modules


def main():

    data_input = data_prep.read_and_prepare_data()

    train_predictors, train_outcome = data_prep.make_training_data(data_input)
    test_names, test_predictors, test_outcome = data_prep.make_test_data(data_input)

    ml_modules.predict_with_knn(train_predictors, train_outcome, test_predictors, test_outcome, test_names)
    ml_modules.predict_with_naive_bayes(train_predictors, train_outcome, test_predictors, test_outcome, test_names)
    ml_modules.predict_with_random_forest(train_predictors, train_outcome, test_predictors, test_outcome)
    ml_modules.predict_with_svm(train_predictors, train_outcome, test_predictors, test_outcome)


if __name__ == '__main__':
        main()
