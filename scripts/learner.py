from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
from processing import plot_validation_curve, plot_learning_curve


def vanilla_fit(estimator, train_x, train_y, test_x, test_y):
    dt = estimator
    dt.fit(train_x, train_y)
    prediction = dt.predict(test_x)
    print(confusion_matrix(test_y, prediction))
    return classification_report(test_y, prediction, output_dict=True)


def do_grid_search(data_name, estimator, parameters, train_x, train_y, test_x, test_y):
    grid_search = GridSearchCV(estimator=estimator,
                               param_grid=[parameters],
                               scoring='f1',
                               cv=KFold(n_splits=5, shuffle=True), n_jobs=-1)

    grid_search.fit(train_x, train_y)
    prediction = grid_search.predict(test_x)

    print("F1 Score: ", metrics.f1_score(test_y, prediction))
    print("Best Params: ", grid_search.best_params_)

    plot_learning_curve(data_name, grid_search.best_estimator_, train_x, train_y, 'f1')
    return classification_report(test_y, prediction, output_dict=True)


def validation_curve(data_name, estimator, parameters, train_x, train_y):
    plot_validation_curve(data_name, estimator, parameters, train_x, train_y, 'f1')
