from PIL import Image
import os
from collections import defaultdict
import numpy as np


def get_named_label(label):
    return {0: "Arno", 1: "Brian", 2: "Daniel", 3: "Huib", 4: "Joop", 5: "Joost S", 6: "Joost W", 7: "Jorn", 8: "Leo", 9: "Wouter"}[label]


def get_label(file_name):
    return int(file_name.split('-')[0]) - 1


def create_labeled_data_set(dir_name):
    data = []
    label_count = defaultdict(int)

    # Convert the images to 2d arrays with the gray values for each pixel (96 x 96 = 9216 pixels)
    files = [(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dir_name)) for f in fn]
    for file in files:
        exercise_file_path = os.path.join(file[0], file[1])
        image = Image.open(exercise_file_path)
        pixels = image.load()
        rows, columns = image.size

        label = get_label(file[1])
        label_count[label] += 1
        data.append((label, []))
        for i in range(rows):
            data[-1][1].append([])
            for j in range(columns):
                data[-1][1][-1].append(pixels[i, j][0])  # Add pixel to data set in a 1d array

    # Split data in training and validation sets
    training_inputs = []
    training_labels = []
    validation_inputs = []
    validation_labels = []
    prev_label = data[0][0]
    tmp_input_set = []
    tmp_label_set = []

    # + 1 because we want the last label to
    for i in range(len(data) + 1):
        if i == len(data):
            cur_label = "END"
            cur_data = []
        else:
            d = data[i]
            cur_label = d[0]
            cur_data = d[1]

        if cur_label != prev_label:
            # shuffle data don't have the same training and validation set every time we call this function
            np.random.shuffle(tmp_input_set)
            np.random.shuffle(tmp_label_set)

            # Split data in training and validation sets
            middle = int(label_count[prev_label] / 2)
            # Training data
            for el in tmp_input_set[middle:]:
                training_labels.append(prev_label)
                training_inputs.append(el)
            # validation data
            for el in tmp_input_set[:middle]:
                validation_labels.append(prev_label)
                validation_inputs.append(el)

            # Reset values
            prev_label = cur_label
            tmp_label_set = []
            tmp_input_set = []

        tmp_label_set.append(cur_label)
        tmp_input_set.append(cur_data)
    # TODO: We should shuffle the data so that the labels aren't ordered like they are now (first all the label 1 then label 2 etc...)
    # TODO: We can't do that with the current implementation because the labels are not in te training inputs
    return (np.array(training_inputs), np.array(training_labels)), (np.array(validation_inputs), np.array(validation_labels))

if __name__ == "__main__":
    (training_inputs, training_labels), (test_inputs, test_labels) = create_labeled_data_set('./images')
    print(len(training_inputs), len(training_labels), len(test_inputs), len(test_labels))
