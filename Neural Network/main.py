import numpy as np
import sys

from net import Net, HyperbolicTangent, Rectifier
sys.path.append('../')  # Move path so that we can import from the datasets directory
from datasets.iris import create_labeled_data_set  # noqa: E402


# Create Data sets
labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
full_set = create_labeled_data_set("../datasets/iris_set.csv")
np.random.shuffle(full_set)
training_data = full_set[:round(len(full_set) - len(full_set) / 3)]
validation_data = full_set[:round(len(full_set) / 3)]

print("Training")
for iris in training_data:
    print(iris)
print()

print("Validation")
for iris in validation_data:
    print(iris)
print()


# Create net work and start training
# After some testing this network seems to perform the best with an 98% correctness on the validation set
iris_classifier = Net([4, 4, 4, 3], HyperbolicTangent, 0.2, 0.3, True)

epocs = 10000
for epoc in range(epocs):
    np.random.shuffle(training_data)
    correct = 0
    for iris in training_data:
        expected_output = [
            float(labels[0] == iris[0]),
            float(labels[1] == iris[0]),
            float(labels[2] == iris[0]),
        ]

        # Train the network
        iris_classifier.feed_forward(iris[1])
        result = iris_classifier.get_results()
        iris_classifier.back_propagate(expected_output)

        # Calculate correctness
        correct += labels[result.index(max(result))] == labels[expected_output.index(max(expected_output))]

    correctness = correct / len(training_data)
    print(epoc / epocs * 100, "%........", correctness)
    # NOTE: If we just let it run for 10'000 epocs we can get up to 98% correctness with the validation data
    # But with an 0.96 on the test set we score 0.95 on average with the validation set with only 50 - 200 epocs
    if correctness >= 0.96:
        print("Program done! in", epoc)
        break

print(iris_classifier)
print()
print()

correct = 0
for iris in validation_data:
    expected_output = [
        float(labels[0] == iris[0]),
        float(labels[1] == iris[0]),
        float(labels[2] == iris[0]),
    ]
    iris_classifier.feed_forward(iris[1])
    result = iris_classifier.get_results()
    correct += result.index(max(result)) == expected_output.index(max(expected_output))

print("Correct: ", correct / len(validation_data) * 100, "%")
