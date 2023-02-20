from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import data_output


def predict_with_knn(train_predictors, train_outcome, test_predictors, test_outcome, test_names):
    knn5 = KNeighborsClassifier(n_neighbors=1)
    knn5.fit(train_predictors, train_outcome)
    y_pred_5 = knn5.predict(test_predictors)

    #data_output.print_prediction_vs_reality(test_names, test_outcome, y_pred_5)
    print("Accuracy knn with k=5", accuracy_score(test_outcome, y_pred_5))


def predict_with_naive_bayes(train_predictors, train_outcome, test_predictors, test_outcome, test_names):
    # Create a Gaussian Classifier
    gnb = GaussianNB()
    # Train the model using the training sets
    gnb.fit(train_predictors, train_outcome)
    # Predict the response for test dataset
    y_pred_nbc = gnb.predict(test_predictors)

    #data_output.print_prediction_vs_reality(test_names, test_outcome, y_pred_nbc)
    print("Accuracy naive bayes", accuracy_score(test_outcome, y_pred_nbc))


def predict_with_random_forest(X_train, y_train, X_test, y_test):
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=70)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy random forest", accuracy_score(y_test, y_pred))


def predict_with_svm(X_train, y_train, X_test, y_test):
    # Create a svm Classifier
    #clf = svm.SVC(kernel='linear'|'rbf'|'sigmoid')
    #clf = svm.SVC()
    clf = svm.SVC(kernel='rbf')

    # Train the model using the training sets
    clf.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy Support vector machines :   ", accuracy_score(y_test, y_pred))



def predict_with_lda(X_train, y_train, X_test, y_test):

    # Create an LDA object
    lda = LinearDiscriminantAnalysis()

    # Fit the LDA model on the training data
    lda.fit(X_train, y_train)

    # Transform the training and testing data using the LDA model
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)

    # Train a classifier on the transformed data
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_lda, y_train)

    # Make predictions on the testing data
    y_pred = clf.predict(X_test_lda)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Linear Discriminant Analysis: {accuracy}")


