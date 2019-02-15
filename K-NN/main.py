import os
from K_NN import calculateK, classify
import sys
sys.path.append('../')  # Move path so that we can import from the datasets directory
from datasets.weather import create_data_set  # noqa: E402


if __name__ == "__main__":
    test_set = create_data_set('../datasets/dataset1.csv')
    validation_set = create_data_set('../datasets/validation1.csv')

    correctness = calculateK(test_set, validation_set)
    # Take the max K
    K = max(correctness, key = lambda x: x[1])[0]
    print()

    unlabeled = [
        (40, 52, 2, 102, 103, 0, 0),
        (25, 48, -18, 105, 72, 6, 1),
        (23, 121, 56, 150, 25, 18, 18),
        (27, 229, 146, 308, 130, 0, 0),
        (41, 65, 27, 123, 95, 0, 0),
        (46, 162, 100, 225, 127, 0, 0),
        (23, -27, -41, -16, 0, 0, -1),
        (28, -78, -106, -39, 67, 0, 0),
        (38, 166, 131, 219, 58, 16, 41),
    ]

    # Label the unlabeled data
    print("Using K:", K)
    for day in unlabeled:
        print(classify(test_set, day, K))

    os.system('pause')
