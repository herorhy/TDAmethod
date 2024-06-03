# max-min采样获取landmark
import numpy as np


def max_min_sampling(points, k):
    """
    :param points: numpy array of shape (n_samples, n_features)
    :param k: number of landmarks to select
    :return: indices of selected landmarks
    """
    # Select the first landmark at random
    n_samples = len(points)
    landmarks = [np.random.randint(0, n_samples)]

    # Calculate the distance from each point to the initial landmark
    distances = np.linalg.norm(points - points[landmarks[0]], axis=1)

    # Select the remaining landmarks iteratively
    for i in range(1, k):
        # Find the point furthest from the existing landmarks
        new_landmark = np.argmax(np.minimum(distances, np.linalg.norm(points - points[landmarks[-1]], axis=1)))
        landmarks.append(new_landmark)

        # Update distances to account for the new landmark
        distances = np.minimum(distances, np.linalg.norm(points - points[new_landmark], axis=1))

    return points[landmarks]
#计算每两个界标点的距离
def pairwise_distances(points):
    n = points.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist[i, j] = np.sqrt(np.sum(np.square(points[i] - points[j])))
            dist[j, i] = dist[i, j]
    return dist
#根据距离选择alphasquare
def select_max_alpha_square(points):
    # Compute the pairwise distances
    dist = pairwise_distances(points)
    # Set the initial value of max_alpha_square to 50% of the maximum distance
    max_alpha_square = ((np.max(dist)**2)/16)*1.2
    print("Initial max_alpha_square:", max_alpha_square)
    return max_alpha_square