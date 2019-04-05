from random import randint, random
from functools import reduce
from operator import add

from evolutionary_algorithm import EvolutionaryAlgorithm


def individual():
    """
    Creates an individual for a population
    An individual has 10 numbered cards from 1 to 10.
    The representation is aan list with the 10 cards,
    0 means the card is in the sum pile, 1 means the card is in the multiply pile
    The cards are sorted from low to high.
    e.g. [0, 0, 0, 1, 1, 1, 0, 0, 0, 0] = (1 + 2 + 3 + 7 + 8 + 9 + 10, 4 * 5* 6) = (30, 120)
    :return: The individual
    """
    return [randint(0, 1) for _ in range(10)]


def fitness(individual, target):
    """
    Determine the fitness of an individual.
    The goal is to have the 10 cards in the individual be divided in 2 piles,
    One with a total sum of all cards in that pile as close to target[0] as possible,
    and one with a multiplication of all cards in the pile as close to target[1] as possible
    The fitness is the error (in this case), so the lower the error the better.

    :param individual: the individual to evaluate
    :param target: The target
    """
    total_sum = 0
    total_multiplication = 1
    for i in range(len(individual)):
        card_value = i + 1
        # Note: We can do this because the piles are represented with booleans, more piles would mean we need some branching
        total_sum += card_value * (not individual[i])
        total_multiplication *= (card_value * individual[i]) or 1
    return abs(target[0] - total_sum) + abs(target[1] - total_multiplication)


def selection_strategy(population_with_fitness, retain_rate, selection_rate):
    graded = [x[1] for x in sorted(population_with_fitness)]
    retain_length = int(len(graded) * retain_rate)
    parents = graded[:retain_length]
    for individual in graded[retain_length:]:
        if selection_rate > random():
            parents.append(individual)

    return parents


def crossover_function(parents, population):
    """
    Crossover function used to create a new generation
    :param parents: The parents
    :param population: The full population (not used for this crossover)
    :return:
    """
    male_index = randint(0, len(parents)-1)
    female_index = randint(0, len(parents)-1)
    while male_index == female_index:
        male_index = randint(0, len(parents)-1)
        female_index = randint(0, len(parents)-1)
    male = parents[male_index]
    female = parents[female_index]
    # child = []
    # for i in range(len(male)):
    #     child.append(male[i] if i % 2 == 0 else female[i])
    half = int(len(male) / 2)
    child = male[:half] + female[half:]
    return child


def mutation_function(individual, population):
    """
    Mutation function:
    Invert bit in the chromosome
    :param individual: The Individual that needs to be mutated
    :param population: The full population (not used for this mutation)
    :return:
    """
    new_individual = individual
    pos_to_mutate = randint(0, len(individual)-1)
    new_individual[pos_to_mutate] = not new_individual[pos_to_mutate]
    return new_individual


if __name__ == "__main__":
    target = (36, 360)  # X
    acceptable_score = 0  # We can stop early when we reached our goal
    p_count = 500  # Number of individuals in population
    generation_count = 500  # Number of generations

    fitness_history = []
    population = None
    """
    After some experimenting with the values for retain_rate, selection_rate, mutation_rate it appear the selection rate
    had the most impact with this problem. With the selection_rate of 0.2 and the other on there default value we can get
    perfect score most of the time.
    """
    test = 0
    total = 1000
    for i in range(total):
        # Create an Evolution object
        EA = EvolutionaryAlgorithm(fitness, selection_strategy, crossover_function, mutation_function, individual)
        # Populate a new generation
        EA.populate(p_count)
        score = 0
        for population, score in EA.evolve(target, 0.2, 0.2):
            fitness_history.append(score)
            if EA.generation >= generation_count or score <= acceptable_score:
                break
        print(score)
        test += (score <= acceptable_score)
    print("Accuracy: ", test / total * 100, '%')  # Avg 90%

    print(fitness_history)
    print(population)
    print(EA.avg_fitness(population, target), "in", EA.generation, "generations")
