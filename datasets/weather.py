import numpy as np


def add_labels(data, labels):
    """
    Adds labels to the given data

    :param data: The data that needs labels
    :param labels: The labels for the data
    :return: A List with Tuples of label and the element in the data List
    """

    # It should not be possible for the sets to have a difference in length, but just in case one of the data sets contains a error d;)
    assert len(labels) == len(data)

    lst = []
    # Add labels to the training data
    for i in range(len(labels)):
        lst.append((labels[i], data[i]))

    return lst


def create_data_set(file_name):
    """
    Create data set from given file.
    This function can only be used for the data set that was given for the exercise

    :param file_name: The file name
    :return: The data set that can be used for the K-NN exercise
    """

    test_data = np.genfromtxt(file_name, delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
    dates = np.genfromtxt(file_name, delimiter=';', usecols=[0])
    test_labels = []
    for label in dates:
        label %= 10000
        if label < 301 or label >= 1201:
            test_labels.append('winter')
        elif 301 <= label < 601:
            test_labels.append('lente')
        elif 601 <= label < 901:
            test_labels.append('zomer')
        elif 901 <= label < 1201:
            test_labels.append('herfst')

    return add_labels(test_data, test_labels)
