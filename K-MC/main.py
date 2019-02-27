import os
from K_MC import calculate_centroids_and_clusters, calculate_k, classify
import sys
sys.path.append('../')  # Move path so that we can import from the datasets directory
from datasets.weather import create_data_set, create_labeled_data_set  # noqa: E402


if __name__ == "__main__":
    unlabeled_data_set = create_data_set('../datasets/dataset1.csv')
    labeled_data_set = create_labeled_data_set('../datasets/dataset1.csv')
    validation_set = create_labeled_data_set('../datasets/dataset1.csv')

    # Label un labeled data using the validation set
    centroids, clusters = calculate_centroids_and_clusters(unlabeled_data_set, calculate_k(unlabeled_data_set))

    # Normally k means clustering is used for unlabelled data,
    # but we have an labeled data set.
    # So we can label our found clusters by comparing the given cluster id with the label it has in de data set.
    # NOTE: It could be that our algorithm finds more (or less) clusters than we have labels, so this is not a 100% validation.

    # Get every label for every cluster id
    label_count_by_id = {}
    for data in labeled_data_set:
        label = classify(centroids, data[1])
        if label in label_count_by_id:
            label_count_by_id[label].append(data[0])
        else:
            label_count_by_id[label] = []

    # Count every label and set the cluster label to the most appearing one
    label_by_id = {}
    for key, labels in label_count_by_id.items():
        count = {}
        for label in labels:
            if label in count:
                count[label] += 1
            else:
                count[label] = 0
        count_max = max(count.items(), key = lambda x: x[1])
        label_by_id[key] = count_max[0]

    # Since we have the labels on our data set, we can also validate our data with our validation set
    correct = 0
    for data in validation_set:
        c = label_by_id[classify(centroids, data[1])]
        correct += (c == data[0])
    print(correct/len(validation_set) * 100)  # AVG: 51~%

    # Test the unlabeled data
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
    for day in unlabeled:
        print(label_by_id[classify(centroids, day)])

    os.system("pause")
