from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import data_output


def predict_with_knn(train_predictors, train_outcome, test_predictors, test_outcome, test_names):
    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn5.fit(train_predictors, train_outcome)
    y_pred_5 = knn5.predict(test_predictors)
    data_output.print_prediction_vs_reality(test_names, test_outcome, y_pred_5)
    print("Accuracy knn with k=5", accuracy_score(test_outcome, y_pred_5))
    print()


def predict_with_naive_bayes(train_predictors, train_outcome, test_predictors, test_outcome, test_names):
    # Create a Gaussian Classifier
    gnb = GaussianNB()
    # Train the model using the training sets
    gnb.fit(train_predictors, train_outcome)
    # Predict the response for test dataset
    y_pred_nbc = gnb.predict(test_predictors)
    data_output.print_prediction_vs_reality(test_names, test_outcome, y_pred_nbc)
    print("Accuracy naive bayes", accuracy_score(test_outcome, y_pred_nbc))
    print()


def predict_with_random_forest(X_train, y_train, X_test, y_test):
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy random forest", accuracy_score(y_test, y_pred))
