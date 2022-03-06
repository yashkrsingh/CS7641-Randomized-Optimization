import itertools

import mlrose_hiive as mlrose

fitness = [mlrose.OneMax(), mlrose.ContinuousPeaks(), mlrose.FlipFlop()]


def tuning_RHC():
    restart_param = -1
    max_fitness = None
    for fn in fitness:
        print("Working to find parameters for {}".format(str(fn)))
        for i in [0, 25, 75, 100]:
            problem = mlrose.DiscreteOpt(length=100, fitness_fn=fn, maximize=True)
            best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem,
                                                                               max_attempts=100,
                                                                               max_iters=100,
                                                                               curve=True,
                                                                               random_state=42,
                                                                               restarts=i)
            if not max_fitness:
                restart_param = i
                max_fitness = best_fitness
            elif best_fitness > max_fitness:
                restart_param = i
                max_fitness = best_fitness
        print("Best RHC 'restart' param for {} = {}".format(fn, str(restart_param)))


def tuning_SA():
    best_param = None
    max_fitness = -1

    decay_params = [[1, 2, 4, 8, 16, 32, 64], [0.01, 0.05, 0.1, 0.15, 0.2, 0.4], [0.001, 0.01, 0.05, 0.1, 1]]

    for fn in fitness:
        print("Working to find parameters for {}".format(str(fn)))
        for i in itertools.product(*decay_params):
            problem = mlrose.DiscreteOpt(length=100, fitness_fn=fn, maximize=True)
            decay = mlrose.GeomDecay(init_temp=i[0], decay=i[1], min_temp=i[2])
            best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem,
                                                                                 max_attempts=100,
                                                                                 max_iters=100,
                                                                                 curve=True,
                                                                                 random_state=42,
                                                                                 schedule=decay)
            if not max_fitness:
                best_param = i
                max_fitness = best_fitness
            elif best_fitness > max_fitness:
                best_param = i
                max_fitness = best_fitness
        print("Best SA 'decay' params for {} = {}".format(fn, str(best_param)))


def tuning_GA():
    best_param = None
    max_fitness = -1

    pop_params = [[100, 200, 400], [0.2, 0.4, 0.8]]

    for fn in fitness:
        print("Working to find parameters for {}".format(str(fn)))
        for i in itertools.product(*pop_params):
            problem = mlrose.DiscreteOpt(length=100, fitness_fn=fn, maximize=True)
            best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem,
                                                                         max_attempts=100,
                                                                         max_iters=100,
                                                                         curve=True,
                                                                         random_state=42,
                                                                         pop_size=i[0],
                                                                         mutation_prob=i[1])
            if not max_fitness:
                best_param = i
                max_fitness = best_fitness
            elif best_fitness > max_fitness:
                best_param = i
                max_fitness = best_fitness
        print("Best GA 'population' params for {} = {}".format(fn, str(best_param)))


def tuning_MIMIC():
    best_param = None
    max_fitness = -1

    for fn in fitness:
        print("Working to find parameters for {}".format(str(fn)))
        for i in [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75]:
            problem = mlrose.DiscreteOpt(length=100, fitness_fn=fn, maximize=True)
            best_state, best_fitness, fitness_curve = mlrose.mimic(problem,
                                                                   max_attempts=100,
                                                                   max_iters=100,
                                                                   curve=True,
                                                                   random_state=42,
                                                                   keep_pct=i)
            if not max_fitness:
                best_param = i
                max_fitness = best_fitness
            elif best_fitness > max_fitness:
                best_param = i
                max_fitness = best_fitness
        print("Best MIMIC 'population' params for {} = {}".format(fn, str(best_param)))


if __name__ == '__main__':
    tuning_RHC()
    tuning_SA()
    tuning_GA()
    tuning_MIMIC()
