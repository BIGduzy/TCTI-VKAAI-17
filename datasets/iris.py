import numpy as np


def map(x, min_x, max_x, min_out, max_out):
    return (x - min_x) * (max_out - min_out) / (max_x - min_x) + min_out


def add_labels(data, labels):
    """
    Adds labels to the given data

    :param data: The data that needs labels
    :param labels: The labels for the data
    :return: A List with Tuples of label and the element in the data List
    """

    # It should not be possible for the sets to have a difference in length, but just in case one of the data sets contains a error d;)
    assert len(labels) == len(data), "Labels and data do not match in size: " + str(len(labels)) + " " + str(len(data))

    lst = []
    # Add labels to the training data
    for i in range(len(labels)):
        lst.append((labels[i][0].decode("utf-8"), data[i]))

    return lst


def create_data_set(file_name):
    """
    Create data set from given file.
    This function can only be used for the data set that was given for the exercise

    :param file_name: The file name
    :return: The data set that can be used for the exercises
    """

    # Summary Statistics:
    #               Min  Max   Mean    SD   Class Correlation
    # sepal length: 4.3  7.9   5.84  0.83    0.7826
    # sepal width:  2.0  4.4   3.05  0.43   -0.4194
    # petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
    # petal width:  0.1  2.5   1.20  0.76    0.9565  (high!)
    # Map data to 0..1
    return np.genfromtxt(file_name, delimiter = ',', usecols = [0, 1, 2, 3],
                         converters={
                             0: lambda s: map(float(s), 4.3, 7.9, 0, 1),
                             1: lambda s: map(float(s), 2.0, 4.4, 0, 1),
                             2: lambda s: map(float(s), 1.0, 6.9, 0, 1),
                             3: lambda s: map(float(s), 0.1, 2.5, 0, 1),
                         }
     )


def create_labeled_data_set(file_name):
    """
    Create data set with labels from given file.
    This function can only be used for the data set that was given for the exercise

    :param file_name: The file name
    :return: The data set that can be used for the exercises
    """
    no_label = create_data_set(file_name)
    labels = np.genfromtxt(file_name, dtype = None, delimiter = ',', usecols = [4], names = True)

    # TODO: Find out why the size of no_label is missing 1
    return add_labels(no_label[:-1], labels)


if __name__ == "__main__":
    print(create_labeled_data_set("iris_set.csv"))
