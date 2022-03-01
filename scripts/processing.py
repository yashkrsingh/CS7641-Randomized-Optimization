import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve


def load_cancer_data():
    cancer = pd.read_csv('../data/breast-cancer-wisconsin.csv', sep=',', header=0)
    cancer.drop(columns=['Sample code number'], inplace=True)
    cancer['Class'] = np.where(cancer['Class'] == 4, 1, 0)
    cancer = cancer[cancer['Bare Nuclei'] != '?']
    # print(cancer['Class'].value_counts())
    return cancer


def load_wine_data():
    wine = pd.read_csv('../data/winequality-white.csv', sep=',', header=0)
    bins = (0, 6, 10)
    labels = [0, 1]
    wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=labels)
    # print(wine['quality'].value_counts())
    return wine


def load_fetus_data():
    fetus = pd.read_csv('../data/fetal-health.csv', sep=',', header=0)
    col_to_drop = ['histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency']
    fetus = fetus.drop(col_to_drop, axis=1)
    fetus["fetal_health"].replace({1: 0, 2: 1, 3: 1}, inplace=True)
    # print(fetus['fetal_health'].value_counts())
    return fetus


def split_data_set(dataframe, seed):
    training_set, test_set = train_test_split(dataframe, train_size=0.8, shuffle=True, random_state=seed)
    train_x, train_y = training_set.iloc[:, :-1], training_set.iloc[:, -1]
    test_x, test_y = test_set.iloc[:, :-1], test_set.iloc[:, -1]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    return train_x, train_y, test_x, test_y


def plot_learning_curve(data_name, estimator, train_x, train_y, score_metric):
    plt.clf()
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(estimator, train_x, train_y, cv=5,
                                                                                    n_jobs=-1, return_times=True,
                                                                                    scoring=score_metric,
                                                                                    train_sizes=np.linspace(0.1, 1.0,
                                                                                                            5))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    score_times_mean = np.mean(score_times, axis=1)
    score_times_std = np.std(score_times, axis=1)

    _, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("F1 Score")
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[0].set_title("Learning curve")

    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-', label="Fit time")
    axes[1].plot(train_sizes, score_times_mean, 'o-', label="Score time")
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].fill_between(train_sizes, score_times_mean - score_times_std, score_times_mean + score_times_std, alpha=0.1)

    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Time (sec)")
    axes[1].legend(loc="best")
    axes[1].set_title("Scalability of the model")

    name = 'learning curve'
    plt.savefig(f'{data_name}_{estimator.__class__.__name__}_{name}.png',
                dpi=200, bbox_inches='tight')


def plot_validation_curve(data_name, estimator, parameters, train_x, train_y, score_metric):
    np.random.seed(42)
    plt.clf()
    n = len(parameters)
    fig, axes = plt.subplots(1, n, figsize=(20, 3))
    for i, param_name in enumerate(parameters):
        param_range = parameters[param_name]
        train_scores, test_scores = validation_curve(estimator, train_x, train_y,
                                                     param_name=param_name, param_range=param_range, n_jobs=-1,
                                                     scoring=score_metric)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        if all([type(p) is tuple for p in param_range]):
            param_range = list(map(str, param_range))

        if all([type(p) is str for p in param_range]):
            print(f"Mapping {param_range} for {param_name}")
            param_labels = param_range
            param_range = list(range(len(param_range)))
        else:
            param_labels = None

        if n > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.grid()
        ax.set_xticks(param_range)
        if param_labels is not None:
            ax.set_xticklabels(labels=param_labels)
        ax.set_xlabel(param_name)
        ax.set_ylabel("Score")
        lw = 2

        ax.plot(param_range, train_scores_mean, label="Training score", color="red", lw=lw)
        ax.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="#DDDDDD", lw=lw)
        ax.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
        ax.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="#DDDDDD", lw=lw)
        ax.legend(loc="best")

    name = 'validation curve'
    plt.savefig(f'{data_name}_{estimator.__class__.__name__}_{name}.png', dpi=200, bbox_inches='tight')


def classification_scores(data, classification_report):
    precision = classification_report['macro avg']['precision']
    recall = classification_report['macro avg']['recall']
    f1 = classification_report['macro avg']['f1-score']
    accuracy = classification_report['accuracy']

    return [data, precision, recall, f1, accuracy]

