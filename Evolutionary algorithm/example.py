from random import randint, random
from functools import reduce
from operator import add

from evolutionary_algorithm import EvolutionaryAlgorithm

"""
The example given in the reader, but refactored to be usable with the EvolutionaryAlgorithm class
"""


def individual(length, min, max):
    """
    Creates an individual for a population
    :param length: the number of values in the list
    :param min: the minimum value in the list of values
    :param max: the maximal value in the list of values
    :return:
    """
    return [randint(min, max) for x in range(length)]


def fitness(individual, target):
    """
    Determine the fitness of an individual.
    Lower is better.
    :param individual: the individual to evaluate
    :param target: the sum of the numbers that we are aiming for (X)"""
    total = reduce(add, individual, 0)
    return abs(target - total)


def selection_strategy(population_with_fitness, retain_rate, selection_rate):
    graded = [x[0] for x in sorted(population_with_fitness, key = lambda x: x[1])]
    retain_length = int(len(graded) * retain_rate)
    parents = graded[:retain_length]
    for individual in graded[retain_length:]:
        if selection_rate > random():
            parents.append(individual)

    return parents


def crossover_function(parents, population):
    male = randint(0, len(parents)-1)
    female = randint(0, len(parents)-1)
    while male == female:
        male = randint(0, len(parents)-1)
        female = randint(0, len(parents)-1)
    male = parents[male]
    female = parents[female]
    half = int(len(male) / 2)
    return male[:half] + female[half:]


def mutation_function(individual, population):
    new_individual = individual
    pos_to_mutate = randint(0, len(individual)-1)
    # this mutation is not ideal, because it
    # restricts the range of possible values,
    # but the function is unaware of the min/max
    # values used to create the individuals
    new_individual[pos_to_mutate] = randint(min(individual), max(individual))
    return new_individual


if __name__ == "__main__":
    target = 333  # X
    p_count = 200  # Number of individuals in population
    generation_count = 200  # Number of generations
    i_length = 7  # N
    i_min = 0  # value range for generating individuals
    i_max = 70

    # Create an Evolution object
    EA = EvolutionaryAlgorithm(fitness, selection_strategy, crossover_function, mutation_function, individual, i_length, i_min, i_max)
    # Populate a new generation
    EA.populate(p_count)

    fitness_history = []
    population = None
    for population, score, best_i in EA.evolve(target, True):
        fitness_history.append(score)
        if EA.generation >= generation_count:
            break

    print(population)
    print(fitness_history)
    print(score)
