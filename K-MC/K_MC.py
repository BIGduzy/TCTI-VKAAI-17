import math
import random
import numpy as np
import matplotlib.pyplot as plt


def calc_distance(attributes1, attributes2):
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


def get_random_index(lst):
    return random.randint(0, len(lst)-1)


def create_random_centroids(data_set, k):
    # create k centroids at random points
    centroids = np.array([data_set[get_random_index(data_set)]])
    for i in range(k - 1):
        data_point = data_set[get_random_index(data_set)]
        while data_point in centroids:
            data_point = data_set[get_random_index(data_set)]

        centroids = np.append(centroids, [data_point], axis = 0)

    return centroids


def cluster_data(centroids, test_set):
    # Cluster data
    clusters = [[] for _ in centroids]
    for data_point in test_set:
        distances = dict()
        # Calculate the distance to every centroid for this data_point
        i = 0
        for centroid in centroids:
            distances[i] = calc_distance(centroid, data_point)
            i += 1

        closest_centroid = min(distances.items(), key = lambda x: x[1])
        clusters[closest_centroid[0]].append(data_point)

    return clusters


def reposition_centroids(clusters):
    new_centroids = []
    for c in clusters:
        # Calculate mean
        new_centroids.append(np.mean(c, axis = 0))

    return new_centroids


def calculate_centroids_and_clusters(data_set, k):
    # print(centroids)
    centroids = create_random_centroids(data_set, k)

    clusters = []
    avg_centroid_move_distance = 999
    centroid_move_diff = avg_centroid_move_distance
    # Reposition centroids and cluster the data_points for new centroids until the move distance of centroids is to small
    while centroid_move_diff > 0.01:
        # Cluster data
        clusters = cluster_data(centroids, data_set)

        # If a cluster is empty start over
        has_empty = True
        while has_empty:
            has_empty = False
            for cluster in clusters:
                if len(cluster) == 0:
                    clusters = cluster_data(centroids, data_set)
                    has_empty = True

        # Centroid repositioning
        old_centroids = centroids
        new_centroids = reposition_centroids(clusters)

        # Calculate the distance moved by
        avg_new_centroid_move_distance = np.mean([calc_distance(old_centroids[i], new_centroids[i]) for i in range(len(old_centroids))])
        centroid_move_diff = abs(avg_centroid_move_distance - avg_new_centroid_move_distance)
        avg_centroid_move_distance = avg_new_centroid_move_distance

        centroids = new_centroids

    return centroids, clusters  # Average distance to centroid


def calculate_total_distance(data_set, k):
    # Calculate centroids and clusters
    centroids, clusters = calculate_centroids_and_clusters(data_set, k)

    # Calculate total within-cluster distance to centroids
    total_distance_to_centroid = 0
    for i in range(len(clusters)):
        centroid = centroids[i]
        cluster = clusters[i]

        for data_point in cluster:
            total_distance_to_centroid += calc_distance(data_point, centroid)  # Average distance to centroid for every centroid

    return total_distance_to_centroid


def calculate_k(data_set):
    # For every k (2 - 9) train the set 10 time and take the best performing result,
    # Then plot the data so that we can pick a k
    # Normally the best performing k is picked by an calculation but we use the 'elbow'
    plt.ion()
    for i in range(1):
        avg = []
        kaas = []
        for k in range(2, 9):
            kaas.append(k)
            multiple = [calculate_total_distance(data_set, k) for _ in range(10)]
            avg.append(np.min(multiple))  # avg total distance to centroid
            print("K:", k, avg[k - 2])
            print(multiple)
        # plot data
        plt.xlabel('Number of clusters')
        plt.ylabel('Total within-cluster distance to centroid')
        plt.plot(kaas, avg)
        plt.show()
        plt.pause(0.001)

    k = int(input("Give your k value: "))
    plt.close()

    return k


def classify(centroids, data_point):
    distances = []
    for centroid in centroids:
        distances.append(calc_distance(centroid, data_point))

    # TODO: Label data
    label_id = 0
    max_dist = distances[label_id]
    for i in range(1, len(distances)):
        d = distances[i]
        if d < max_dist:
            max_dist = d
            label_id = i

    return label_id


if __name__ == "__main__":
    print("K - MC")
