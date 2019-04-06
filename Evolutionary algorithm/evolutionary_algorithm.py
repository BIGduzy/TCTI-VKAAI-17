from random import randint, random
from functools import reduce
from operator import add


class EvolutionaryAlgorithm:
    def __init__(self, fitness_function, selection_strategy, crossover_function, mutation_function, individual, *individual_args):
        """
        Constructor for EvolutionaryAlgorithm.

        :param fitness_function:
        The fitness function
        (Requires the "(individual (individual), target (any)): return score (any)" function definition)
        :param selection_strategy:
        The selection strategy
        (Requires the "(population_with_fitness (tuple), retain_rate (float), selection_rate(float)): parents (list)" function definition)
        :param crossover_function:
        The crossover function
        (Requires the "(parents (list), population (list)): return new_individual (individual)" function definition)
        :param mutation_function:
        The mutation function
        (Requires the "(individual (individual), population (list)): return mutated_individual (individual)" function definition)
        :param individual: The function used for creating an individual (can be a class)
        :param individual_args: All arguments necessary for the individual function
        """
        self.fitness_function = fitness_function
        self.selection_strategy = selection_strategy
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.individual = individual
        self.individual_args = individual_args

        self.generation = 0
        self.population = []
        self.population_with_fitness = []

    def _evolution_iteration(self, target, retain_rate = 0.2, selection_rate = 0.05, mutation_rate = 0.01):
        """
        Internal function not to be used by the user (use the evolution function instead).
        Uses the given functions and strategy to create one generation for the algorithm.

        :param target: The target for the algorithm (directly passed to the user defined fitness function)
        :param retain_rate: The retain rate used by the user defined selection strategy
        :param selection_rate: The selection rate for each individual used by the user defined selection strategy
        :param mutation_rate: The mutation rate, used by the user defined mutation function
        :return:
        """
        # Calculate the fitness for each individual in the population (and pair them with the individual)
        self.population_with_fitness = [(x, self.fitness_function(x, target)) for x in self.population]
        # Select the parents we want to use for crossover
        parents = self.selection_strategy(self.population_with_fitness, retain_rate, selection_rate)

        # Crossover parents to create off spring
        desired_length = len(self.population) - len(parents)
        children = []
        while len(children) < desired_length:
            children.append(self.crossover_function(parents, self.population))

        # mutate some individuals
        for i in range(len(children)):
            if mutation_rate > random():
                children[i] = self.mutation_function(children[i], self.population)

        parents.extend(children)
        return parents

    def avg_fitness(self):
        """
        Find average fitness for a population
        """
        summed = reduce(add, (x[1] for x in self.population_with_fitness), 0)
        return summed / len(self.population_with_fitness)

    def get_best_individual(self, score_is_error):
        """
        Find best individual in population
        :return: A tuple with (individual, its score)
        """
        best = self.population_with_fitness[0]
        for i in self.population_with_fitness:
            if (score_is_error and i[1] < best[1]) or (not score_is_error and i[1] > best[1]):
                best = i

        return best

    def populate(self, population_count = 100):
        """
        Create a number of individuals (i.e., a population).
        :param population_count: The desired size of the population
        """
        self.population = [self.individual(*self.individual_args) for _ in range(population_count)]

    def evolve(self, target, score_is_error , retain_rate = 0.2, selection_rate = 0.05, mutation_rate = 0.01):
        """
        The Evolution algorithm
        Yields each generation of the evolution

        :param target: The target for the algorithm (directly passed to the user defined fitness function)
        :param retain_rate: The retain rate used by the user defined selection strategy
        :param selection_rate: The selection rate for each individual used by the user defined selection strategy
        :param mutation_rate: The mutation rate, used by the user defined mutation function
        :return:
        """
        assert len(self.population) != 0, "Generation has not yet been populated (use self.populate)"
        while True:
            self.population = self._evolution_iteration(target, retain_rate, selection_rate, mutation_rate)
            self.generation += 1

            yield self.population, self.avg_fitness(), self.get_best_individual(score_is_error)
