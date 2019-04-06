import sys
import os
import tensorflow as tf
import time
import pygame
import pygame.freetype  # Import the freetype module.
from random import randint, random

from evolutionary_algorithm import EvolutionaryAlgorithm
sys.path.append('../')  # Move path so that we can import from the datasets directory
from datasets.faces import create_labeled_data_set, get_named_label  # noqa: E402

# Disable tf prints
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


# GLOBALS
max_neurons = 600
max_epochs = 400

def individual():
    """
    Creates an individual for a population
    An individual has 4 parameters:
    number of neurons in first hidden layer (1, max_neurons)
    number of neurons in second hidden layer (1, max_neurons)
    Dropout rate (0 ... 1)
    Number of epochs (1, max_epochs)
    :return: The individual
    """
    global max_epochs
    global max_neurons
    return [randint(1, max_neurons), randint(1, max_neurons), random(), randint(1, max_epochs)]


def fitness(individual, data):
    """
    Determine the fitness of an individual.
    Creates a network and trains it, then evaluates the model with test data.
    The goal is to create a model with a high as possible accuracy in as little as possible epochs,
    Score: 0 - 100%
    100% is perfect

    :param individual: the individual to evaluate
    :param data: The data used to train and validate the network
    """
    print("CALCULATING FITNESS")
    print("Hidden layer 1: ", individual[0])
    print("Hidden layer 2: ", individual[1])
    print("Dropout: ", individual[2])
    print("Epochs: ", individual[3])

    (training_inputs, training_labels), (test_inputs, test_labels) = data
    # Create the neural network
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(96, 96)),
      tf.keras.layers.Dense(individual[0], activation=tf.nn.relu),
      tf.keras.layers.Dense(individual[1], activation=tf.nn.relu),
      tf.keras.layers.Dropout(individual[2]),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model (We have a small data set so we need a lot of epochs)
    debug = False
    model.fit(training_inputs, training_labels, epochs=individual[3], verbose = debug)

    # Validate the model
    result = model.evaluate(test_inputs, test_labels)
    score = result[1] * 100 - individual[3] / max_epochs * 10
    print(score)
    print()

    return score


def selection_strategy(population_with_fitness, retain_rate, selection_rate):
    # TODO: We need a better selection strategy
    graded = [x[0] for x in sorted(population_with_fitness, key = lambda x: x[1], reverse = True)]
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
    global max_epochs
    global max_neurons
    new_individual = individual
    pos_to_mutate = randint(0, len(individual)-1)

    if pos_to_mutate == 0 or pos_to_mutate == 1:
        new_val = randint(1, max_neurons)
    elif pos_to_mutate == 2:
        new_val = random()
    else:
        new_val = randint(1, max_epochs)

    new_individual[pos_to_mutate] = new_val
    return new_individual

if __name__ == "__main__":
    # Create data set
    (training_inputs, training_labels), (test_inputs, test_labels) = create_labeled_data_set("../datasets/images")
    # Normalize data
    training_inputs, test_inputs = training_inputs / 255.0, test_inputs / 255.0

    p_count = 12
    target = ((training_inputs, training_labels), (test_inputs, test_labels))
    fitness_history = []
    generation_count = 4
    acceptable_score = 100

    # Create an Evolution object
    EA = EvolutionaryAlgorithm(fitness, selection_strategy, crossover_function, mutation_function, individual)
    # Populate a new generation
    EA.populate(p_count)
    score = 0
    print("POPULATION COUNT: ", len(EA.population))
    print("STARTING GENERATION")
    print()
    for population, score, best_individual in EA.evolve(target, False, 0.2, 0.5):
        fitness_history.append(score)
        print("GENERATION SCORE: ")
        print("Gen: ", EA.generation, "avg score:", score, "Best score:", best_individual[1])
        print()
        if EA.generation >= generation_count or score >= acceptable_score:
            break
        print("STARTING NEW GENERATION")


    # TODO: Save Best of generation

    # TODO: Get Best of generatoin
    individual = best_individual[0]
    # Create the neural network
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(96, 96)),
      tf.keras.layers.Dense(individual[0], activation=tf.nn.relu),
      tf.keras.layers.Dense(individual[1], activation=tf.nn.relu),
      tf.keras.layers.Dropout(individual[2]),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model (We have a small data set so we need a lot of epochs)
    model.fit(training_inputs, training_labels, epochs=individual[3], verbose = 0)

    # Validate the model
    result = model.evaluate(test_inputs, test_labels)
    print("individual: ", individual)
    print("Result: ", result)

    # Use the network / Render the faces
    pygame.init()
    pygame.font.init()
    myfont = pygame.freetype.SysFont('Comic Sans MS', 30)
    tile_size = 10
    screen = pygame.display.set_mode((tile_size * 96, tile_size * 96))

    running = True
    index = 0
    predictions = model.predict(test_inputs)
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Draw number
            test_input = test_inputs[index]
            test_label = test_labels[index]
            prediction = predictions[index]
            for i in range(len(test_input)):
                row = test_input[i]
                for j in range(len(row)):
                    pixel = test_input[i][j]
                    color = (pixel * 255, pixel * 255, pixel * 255)
                    pygame.draw.rect(screen, color, pygame.Rect(i * tile_size, j * tile_size, tile_size, tile_size))

            index += 1
            index %= len(test_inputs)
            # Draw label of the number and what the model predicted
            myfont.render_to(screen, (5, 5), "Prediction: " + get_named_label(prediction.argmax()), (0, 0, 0))
            myfont.render_to(screen, (5, 50), "Actual: " + get_named_label(test_label), (0, 0, 0))

            # Display everything
            pygame.display.flip()

            # Small delay
            time.sleep(1)
        pygame.quit()
    except SystemExit:
        pygame.quit()
