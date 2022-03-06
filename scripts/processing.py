from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve


def load_wine_data():
    wine = pd.read_csv('../data/winequality-white.csv', sep=',', header=0)
    bins = (0, 6, 10)
    labels = [0, 1]
    wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=labels)
    # print(wine['quality'].value_counts())
    return wine


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
    plt.savefig(f'{data_name}_{estimator.__class__.__name__}_{name}.png', dpi=200, bbox_inches='tight')


def plot_comparison_bar(dataframe):
    plt.clf()

    one_max_df = dataframe[0:4]
    cont_peak_df = dataframe[4:8]
    flip_flop_df = dataframe[8:12]
    df_list = [one_max_df, cont_peak_df, flip_flop_df]
    name_list = ['One Max', 'Continuous Peaks', 'Flip Flop']

    for i in range(3):
        fig, axes = plt.subplots(2, 2, figsize=(20, 5))
        axes[0][0].set_ylabel("Fitness")
        axes[0][0].grid()
        axes[0][0].bar(x='algo', height='fitness', data=df_list[i], width=0.3)
        axes[0][0].set_title("Fitness")

        axes[0][1].set_ylim(0, 8)
        axes[0][1].set_ylabel("Time (in seconds)")
        axes[0][1].grid()
        axes[0][1].bar(x='algo', height='time', data=df_list[i], width=0.3)
        axes[0][1].set_title("Time Taken")

        axes[1][0].set_ylim(0, 1500)
        axes[1][0].set_ylabel("Number of Iterations")
        axes[1][0].grid()
        axes[1][0].bar(x='algo', height='iterations', data=df_list[i], width=0.3)
        axes[1][0].set_title("Iterations")

        axes[1][1].set_ylim(0, 50000)
        axes[1][1].set_ylabel("Number of FEvals")
        axes[1][1].grid()
        axes[1][1].bar(x='algo', height='fevals', data=df_list[i], width=0.3)
        axes[1][1].set_title("Function Evaluations")

        fig.suptitle(f'Runtime Statistics for {name_list[i]}')
        plt.savefig(f'Stats_{name_list[i]}.png', dpi=200, bbox_inches='tight')


def classification_scores(data, classification_report):
    precision = classification_report['macro avg']['precision']
    recall = classification_report['macro avg']['recall']
    f1 = classification_report['macro avg']['f1-score']
    accuracy = classification_report['accuracy']

    return [data, precision, recall, f1, accuracy]


def run_randomized_optimizer(name, problem, results, seed, restart, i_temp, decay, m_temp, pop_size, mprob, keep_pct):
    curves = {}
    fit = []

    start = time.perf_counter()
    decay = mlrose.GeomDecay(init_temp=i_temp, decay=decay, min_temp=m_temp)
    best_state, best_fitness, curve = mlrose.simulated_annealing(problem, schedule=decay, curve=True, random_state=seed,
                                                                 max_attempts=100)
    tt = time.perf_counter() - start
    results.loc[results.shape[0]] = [name, 'sa', best_fitness, len(curve), problem.fitness_evaluations, tt]
    curve = deepcopy(curve)
    curves['sa'] = curve
    fit.append(best_fitness)

    start = time.perf_counter()
    best_state, best_fitness, curve = mlrose.random_hill_climb(problem, restarts=restart, curve=True, random_state=seed,
                                                               max_attempts=100)
    tt = time.perf_counter() - start
    results.loc[results.shape[0]] = [name, 'rhc', best_fitness, len(curve), problem.fitness_evaluations, tt]
    curve = deepcopy(curve)
    curves['rhc'] = curve
    fit.append(best_fitness)

    start = time.perf_counter()
    best_state, best_fitness, curve = mlrose.genetic_alg(problem, pop_size=pop_size, mutation_prob=mprob, curve=True,
                                                         random_state=seed, max_attempts=100)
    tt = time.perf_counter() - start
    results.loc[results.shape[0]] = [name, 'ga', best_fitness, len(curve), problem.fitness_evaluations, tt]
    curve = deepcopy(curve)
    curves['ga'] = curve
    fit.append(best_fitness)

    start = time.perf_counter()
    best_state, best_fitness, curve = mlrose.mimic(problem, keep_pct=keep_pct, curve=True, random_state=seed,
                                                   max_attempts=100)
    tt = time.perf_counter() - start
    results.loc[results.shape[0]] = [name, 'mimic', best_fitness, len(curve), problem.fitness_evaluations, tt]
    curve = deepcopy(curve)
    curves['mimic'] = curve
    fit.append(best_fitness)

    return results, curves, max(fit)


def normalize(arr, max=None):
    arr = np.asarray(arr)
    if max is None:
        arr -= arr.min()
        arr /= arr.max()
    else:
        arr /= max
    return arr


def evaluate_random_algorithms(name, curve_dict, max, is_nn=False):
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    ylabel = 'Loss' if is_nn else 'Fitness'
    axes[0].set_xlabel("Iterations%")
    axes[0].set_ylabel(f"{ylabel}%")
    axes[0].grid()

    axes[1].set_xlabel("Fevals%")
    axes[1].set_ylabel(f"{ylabel}%")
    axes[1].grid()

    for alg, curve in curve_dict.items():
        fitness_by_iterations = normalize([c[0] for c in curve], max)
        fevals_by_iterations = normalize([c[1] for c in curve])

        iterations = normalize(list(map(float, range(len(curve)))))

        axes[0].plot(iterations, fitness_by_iterations, label=alg)
        axes[1].plot(fevals_by_iterations, fitness_by_iterations, label=alg)

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    title_name = name
    if is_nn:
        title_name = "Neural Net Optimization"
    fig.suptitle(f'Comparison of Normalized Characteristics for {title_name}')
    plt.savefig(f'Compare_{name}.png', dpi=200, bbox_inches='tight')
