import matplotlib.pyplot as plt
import data_prep


def draw_plots(X_test, y_pred_5, y_pred_1, X, y):
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='*', s=100, edgecolors='black')
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_5, marker='*', s=100, edgecolors='black')
    plt.title("Predicted values with k=5", fontsize=20)

    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_1, marker='*', s=100, edgecolors='black')
    plt.title("Predicted values with k=1", fontsize=20)
    plt.show()


def print_prediction_vs_reality(test_names, test_predictions, test_outcomes):
    tn = test_names.tolist()
    tp = test_predictions.tolist()
    to = test_outcomes.tolist()
    dict_names = data_prep.make_category_dict()
    print(f'{"name":30} {"category":20} {"predicted":20} {"outcome":10}')
    print("--------------------------------------------------------------------------------")
    count = 0
    while count < len(test_outcomes):
        if tp[count] == to[count]:
            pred = "correct"
        else:
            pred = "wrong"
        real_name = dict_names[tp[count]]
        predicted_name = dict_names[to[count]]
        print(f'{tn[count]:30} {real_name:20} {predicted_name:20} {pred:10}' )
        count+=1