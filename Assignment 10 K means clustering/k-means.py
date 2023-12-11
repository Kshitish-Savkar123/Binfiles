import csv
import random
from math import sqrt

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            data.append([float(row[0]), float(row[1])])
    return data

def initialize_centroids(data, k):
    return random.sample(data, k)

def euclidean_distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def assign_to_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]

    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        min_distance_index = distances.index(min(distances))
        clusters[min_distance_index].append(point)

    return clusters

def update_centroids(clusters):
    centroids = []

    for cluster in clusters:
        if cluster:
            mean_x = sum(point[0] for point in cluster) / len(cluster)
            mean_y = sum(point[1] for point in cluster) / len(cluster)
            centroids.append([mean_x, mean_y])
        else:
            # If a cluster is empty, keep the centroid unchanged
            centroids.append([0, 0])

    return centroids

def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)

    for _ in range(max_iterations):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(clusters)

        # Check for convergence
        if new_centroids == centroids:
            break

        centroids = new_centroids

    return clusters, centroids

if __name__ == "__main__":
    filename = 'data.csv'  # Replace with the actual file name
    k = 3  # Number of clusters

    data = read_csv(filename)
    clusters, centroids = k_means(data, k)

    # Print the results
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")

    print("Final Centroids:")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i + 1}: {centroid}")

# import pandas as pd
# import numpy as np

# def calculate_distance(point1, point2):
#     return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5

# def kmeans_clustering(data, num_clusters, max_iterations=100):
#     # Initialize centroids randomly
#     centroids = data.sample(n=num_clusters, random_state=42)
#     centroids = centroids.values

#     for _ in range(max_iterations):
#         # Assign each data point to the nearest centroid
#         data['cluster'] = data.apply(
#             lambda row: min(range(num_clusters), key=lambda i: calculate_distance(centroids[i], row)),
#             axis=1
#         )

#         # Update centroids based on the mean of assigned points
#         new_centroids = [data[data['cluster'] == i][['feature1', 'feature2']].mean().values for i in range(num_clusters)]

#         # Check for convergence
#         if np.array_equal(centroids, new_centroids):
#             break

#         centroids = new_centroids

#     return data, centroids

# # Read the CSV file
# input_file_path = 'data.csv'  # Replace with your CSV file path
# df = pd.read_csv(input_file_path)

# # Assuming your data has features 'feature1' and 'feature2', modify this according to your data
# features = df[['feature1', 'feature2']]

# # Get the number of clusters from the user
# num_clusters = int(input("Enter the number of clusters: "))

# # Perform K-means clustering
# result_df, cluster_centers = kmeans_clustering(features, num_clusters)

# # Print cluster centers
# print("\nCluster Centers:")
# for i, center in enumerate(cluster_centers):
#     print(f"Cluster {i + 1}: {center}")

# # Save the clustered data to a new CSV file if needed
# output_file_path = 'output_clustered_data.csv'
# result_df.to_csv(output_file_path, index=False)
# print(f"\nClustered data saved to {output_file_path}")


# # With 3 features
# import pandas as pd
# import numpy as np

# def calculate_distance(point1, point2):
#     return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5

# def kmeans_clustering(data, num_clusters, max_iterations=100):
#     # Initialize centroids randomly
#     centroids = data.sample(n=num_clusters, random_state=42)
#     centroids = centroids.values

#     for _ in range(max_iterations):
#         # Assign each data point to the nearest centroid
#         data['cluster'] = data.apply(
#             lambda row: min(range(num_clusters), key=lambda i: calculate_distance(centroids[i], row)),
#             axis=1
#         )

#         # Update centroids based on the mean of assigned points
#         new_centroids = [data[data['cluster'] == i][['feature1', 'feature2', 'feature3']].mean().values for i in range(num_clusters)]

#         # Check for convergence
#         if np.array_equal(centroids, new_centroids):
#             break

#         centroids = new_centroids

#     return data, centroids

# # Read the CSV file
# input_file_path = 'data.csv'  # Replace with your CSV file path
# df = pd.read_csv(input_file_path)

# # Assuming your data has features 'feature1', 'feature2', and 'feature3', modify this according to your data
# features = df[['feature1', 'feature2', 'feature3']]

# # Get the number of clusters from the user
# num_clusters = int(input("Enter the number of clusters: "))

# # Perform K-means clustering
# result_df, cluster_centers = kmeans_clustering(features, num_clusters)

# # Print cluster centers
# print("\nCluster Centers:")
# for i, center in enumerate(cluster_centers):
#     print(f"Cluster {i + 1}: {center}")

# # Save the clustered data to a new CSV file if needed
# output_file_path = 'output_clustered_data.csv'
# result_df.to_csv(output_file_path, index=False)
# print(f"\nClustered data saved to {output_file_path}")


# import pandas as pd
# import numpy as np

# def calculate_distance(point1, point2):
#     return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5

# def kmeans_clustering(data, num_clusters, max_iterations=100):
#     # Initialize centroids randomly
#     centroids = data.sample(n=num_clusters, random_state=42)
#     centroids = centroids.values

#     for _ in range(max_iterations):
#         # Assign each data point to the nearest centroid and calculate distances
#         data['cluster'] = data.apply(
#             lambda row: min(range(num_clusters), key=lambda i: calculate_distance(centroids[i], row)),
#             axis=1
#         )
#         data['distance_to_centroid'] = data.apply(
#             lambda row: calculate_distance(centroids[int(row['cluster'])], row),
#             axis=1
#         )

#         # Update centroids based on the mean of assigned points
#         new_centroids = [data[data['cluster'] == i][['feature1', 'feature2']].mean().values for i in range(num_clusters)]

#         # Check for convergence
#         if np.array_equal(centroids, new_centroids):
#             break

#         centroids = new_centroids

#     # Calculate cluster radius
#     cluster_radii = data.groupby('cluster')['distance_to_centroid'].max().to_dict()

#     return data, centroids, cluster_radii

# # Read the CSV file
# input_file_path = 'data.csv'  # Replace with your CSV file path
# df = pd.read_csv(input_file_path)

# # Assuming your data has features 'feature1' and 'feature2', modify this according to your data
# features = df[['feature1', 'feature2']]

# # Get the number of clusters from the user
# num_clusters = int(input("Enter the number of clusters: "))

# # Perform K-means clustering
# result_df, cluster_centers, cluster_radii = kmeans_clustering(features, num_clusters)

# # Print cluster centers and radii
# print("\nCluster Centers and Radii:")
# for i, (center, radius) in enumerate(zip(cluster_centers, cluster_radii.values())):
#     print(f"Cluster {i + 1}: Center {center}, Radius {radius:.2f}")

# # Save the clustered data to a new CSV file if needed
# output_file_path = 'output_clustered_data.csv'
# result_df.to_csv(output_file_path, index=False)
# print(f"\nClustered data saved to {output_file_path}")