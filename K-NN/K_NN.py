import math
import os

def distance(attributes1, attributes2):
    """
    Calculate euclidean distance between two neighbors

    :param attributes1: Attributes of neighbor1
    :param attributes2: Attributes of neighbor2
    :return: The distance between the neighbors
    """
    # It should not be possible for the sets to have a difference in length, but just in case one of the data sets contains a error d;)
    assert len(attributes1) == len(attributes2)

    distance = 0
    for i in range(len(attributes1)):
        distance += pow(attributes1[i] - attributes2[i], 2)
    return math.sqrt(distance)


def get_neighbors(test_set, data_point, k):
    """
    Get the K neighbors for given data_point

    :param test_set: The test set
    :param data_point: The data_point that is used for finding its neigbors
    :param k: Number of neighbors that need to be returned
    :return: List with neighbors
    """
    assert k < len(test_set)

    # Calculate distances to the other data_points
    distances = []
    for other_data_point in test_set:
        distances.append((other_data_point[0], distance(other_data_point[1], data_point)))

    # sort the distances from short to long
    distances.sort(key = lambda x: x[1])

    # return the first K neighbors
    return distances[:k]


def classify(test_set, data_point, k):
    """
    Returns the classification based on most common label in the neighbors list

    :param test_set: The test set
    :param data_point: The unlabeled data_point that needs to be classified
    :param k: Number of neighbors that need to be returned
    :return: The classification
    """

    # Get the neighbors
    neighbors = get_neighbors(test_set, data_point, k)

    # Count all labels in neighbors list
    count = {}
    for neighbor in neighbors:
        label = neighbor[0]
        if label in count:
            count[label] += 1
        else:
            count[label] = 0

    # Return most common label in neighbors list
    return max(count.items(), key = lambda x: x[1])[0]


def calculateK(test_set, validation_set, max_k = 100, print_progress = True):
    """
    Calculate the K for the K Nearest Neighbor using the test_set and a validation_set

    :param test_set: The "training" data for the K Nearest Neighbor implementation
    :param validation_set: The validation data for the K Nearest Neighbor implementation
    :param max_k: The maximum K value
    :param print_progress: If the function needs to print its current progress (default: True)
    :return: A Tuple of
    """

    # Calculate the best K value
    correctness = []
    for K in range(1, max_k):
        correct = 0
        for data_point in validation_set:
            result = classify(test_set, data_point[1], K)

            # Compare the classification result with the expected label
            if result == data_point[0]:
                correct += 1
        correctness.append((K, float(correct) / float(len(validation_set)) * 100))

        # Clear the console an print the current K and its correctness
        if print_progress:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(correctness[-1])

    return correctness
