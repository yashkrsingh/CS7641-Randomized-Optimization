from sklearn.metrics import classification_report

from processing import *


def part1():
    seed = 42
    PRB_LEN = 100
    np.random.seed(seed)

    results = pd.DataFrame(columns=['problem', 'algo', 'fitness', 'iterations', 'fevals', 'time'])

    # One Max - SA
    one_max_fitness = mlrose.OneMax()
    one_max_problem = mlrose.DiscreteOpt(length=PRB_LEN, fitness_fn=one_max_fitness, maximize=True, max_val=2)
    results, curves, best_fit = run_randomized_optimizer('one-max', one_max_problem, results, seed, 100, 1, 0.1, 0.001, 100, 0.05, 0.15)
    evaluate_random_algorithms('One Max', curves, best_fit)

    # Continuous Peaks - GA
    cont_peak_fitness = mlrose.ContinuousPeaks(t_pct=0.15)
    cont_peak_problem = mlrose.DiscreteOpt(length=PRB_LEN, fitness_fn=cont_peak_fitness, maximize=True, max_val=2)
    results, curves, best_fit = run_randomized_optimizer('cont-peaks', cont_peak_problem, results, seed, 200, 1, 0.1, 1, 200, 0.4, 0.2)
    evaluate_random_algorithms('Continuous Peaks', curves, best_fit)

    # Four Peaks - GA
    # four_peaks_fitness = mlrose.FourPeaks()
    # four_peaks_problem = mlrose.DiscreteOpt(length=PRB_LEN, fitness_fn=four_peaks_fitness, maximize=True, max_val=2)
    # results, curves, best_fit = run_randomized_optimizer('four-peaks', four_peaks_problem, results, seed, 200, 1, 0.1, 1, 200, 0.4, 0.2)
    # evaluate_random_algorithms('Four Peaks', curves, best_fit)

    # Flip Flop - MIMIC
    flip_flop_fitness = mlrose.FlipFlop()
    flip_flop_problem = mlrose.DiscreteOpt(length=PRB_LEN, fitness_fn=flip_flop_fitness, maximize=True, max_val=2)
    results, curves, best_fit = run_randomized_optimizer('flip-flop', flip_flop_problem, results, seed, 100, 1, 0.1, 1, 200, 0.2, 0.2)
    evaluate_random_algorithms('Flip Flop', curves, best_fit)

    results.to_csv('results_optimize.csv', sep=',', encoding='utf-8')
    plot_comparison_bar(results)


def part2():
    seed = 42
    np.random.seed(seed)

    results = pd.DataFrame(columns=['data', 'precision', 'recall', 'f1', 'accuracy'])
    curves = {}
    loss = -1

    wine = load_wine_data()
    wine_train_x, wine_train_y, wine_test_x, wine_test_y = split_data_set(wine, seed)

    # Random Hill Climbing
    nn_rhc = mlrose.NeuralNetwork(hidden_nodes=[200], curve=True, random_state=seed,
                                  clip_max=5,
                                  learning_rate=0.0001, early_stopping=True,
                                  max_iters=10000, max_attempts=50)
    nn_rhc.fit(wine_train_x, wine_train_y)
    pred_rhc = nn_rhc.predict(wine_test_x)
    test_rhc = classification_report(wine_test_y, pred_rhc, output_dict=True)
    results.loc[results.shape[0]] = classification_scores('rhc', test_rhc)
    plot_learning_curve('rhc', nn_rhc, wine_train_x, wine_train_y, 'f1')
    curve = nn_rhc.fitness_curve
    for c in curve:
        if c[0] > loss:
            loss = c[0]
    curve = deepcopy(curve)
    curves['rhc'] = curve

    # Simulated Annealing
    nn_sa = mlrose.NeuralNetwork(hidden_nodes=[200], algorithm='simulated_annealing', curve=True, random_state=seed,
                                 clip_max=5,
                                 learning_rate=0.0001, early_stopping=True,
                                 max_iters=10000, max_attempts=50)
    nn_sa.fit(wine_train_x, wine_train_y)
    pred_sa = nn_sa.predict(wine_test_x)
    test_sa = classification_report(wine_test_y, pred_sa, output_dict=True)
    results.loc[results.shape[0]] = classification_scores('sa', test_sa)
    plot_learning_curve('sa', nn_sa, wine_train_x, wine_train_y, 'f1')
    curve = nn_sa.fitness_curve
    for c in curve:
        if c[0] > loss:
            loss = c[0]
    curve = deepcopy(curve)
    curves['sa'] = curve

    # Genetic Algorithm
    nn_ga = mlrose.NeuralNetwork(hidden_nodes=[200], algorithm='genetic_alg', curve=True, random_state=seed,
                                 clip_max=5,
                                 learning_rate=0.0001, early_stopping=True,
                                 max_iters=10000, max_attempts=50, pop_size=10)
    nn_ga.fit(wine_train_x, wine_train_y)
    pred_ga = nn_ga.predict(wine_test_x)
    test_ga = classification_report(wine_test_y, pred_ga, output_dict=True)
    results.loc[results.shape[0]] = classification_scores('ga', test_ga)
    plot_learning_curve('ga', nn_ga, wine_train_x, wine_train_y, 'f1')
    curve = nn_ga.fitness_curve
    for c in curve:
        if c[0] > loss:
            loss = c[0]
    curve = deepcopy(curve)
    curves['ga'] = curve

    evaluate_random_algorithms('Neural Network', curves, loss, is_nn=True)
    results.to_csv('results_nn.csv', sep=',', encoding='utf-8')


if __name__ == '__main__':
    # part1()
    part2()

