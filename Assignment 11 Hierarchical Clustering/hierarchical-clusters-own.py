# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import dendrogram, linkage
# from scipy.spatial.distance import squareform
# import pandas as pd

# # Read distance matrix from CSV file
# csv_file = 'linkage_input.csv'
# df = pd.read_csv(csv_file, index_col=0)

# # Convert DataFrame to NumPy array
# distance_matrix = df.to_numpy()

# # Convert to condensed distance matrix
# condensed_distance_matrix = squareform(distance_matrix)

# # Hierarchical clustering with single linkage
# linkage_matrix = linkage(condensed_distance_matrix, method='single')

# # Plot the dendrogram
# dendrogram(linkage_matrix, labels=df.index)
# plt.title('Hierarchical Clustering Dendrogram (Single Linkage)')
# plt.xlabel('Data Points')
# plt.ylabel('Distance')
# plt.show()

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def read_distance_matrix(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        labels = []
        distances = []
        for row in reader:
            labels.append(row[0])
            distances.append([float(val) for val in row[1:]])
    return labels, np.array(distances)

def single_linkage(cluster1, cluster2, distances):
    min_distance = float('inf')
    for i in cluster1:
        for j in cluster2:
            min_distance = min(min_distance, distances[i, j])
    return min_distance

def hierarchical_clustering(distances):
    clusters = [[i] for i in range(len(distances))]
    iteration = 1
    while len(clusters) > 1:
        min_distance = float('inf')
        merge_indices = (0, 0)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = single_linkage(clusters[i], clusters[j], distances)
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)
        i, j = merge_indices
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
        print(f"Iteration {iteration}: {clusters}")

        # Print the distance matrix for the current iteration
        print(f"Distance Matrix (Iteration {iteration}):")
        print(distances[np.ix_(clusters[i], clusters[i])])

        iteration += 1
    return clusters[0]

if __name__ == "__main__":
    file_path = 'linkage_input.csv'  # Replace with your CSV file path
    labels, distances = read_distance_matrix(file_path)
    cluster = hierarchical_clustering(distances)