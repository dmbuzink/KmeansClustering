from typing import List, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs 
from sklearn.cluster import kmeans_plusplus   
from sklearn.metrics.pairwise import euclidean_distances

from pyspark import SparkConf, SparkContext, RDD

from Vertex import Vertex

spark: SparkContext = None

def create_blobs() -> Tuple[np.ndarray, np.ndarray]:
    n_samples = 100
    n_clusters = 3

    return make_blobs(n_samples=n_samples, random_state=8, centers=n_clusters)

def perform_clustering(P: np.ndarray, k: int, epsilon: float = 0.1):

    global spark
    
    # Create vertices for points
    vertices = [Vertex(x, y, 1) for (x, y) in P]
    # Add keys to points
    vertices_with_index = [(i, vertices[i]) for i in range(len(vertices))]
    # Parallelize
    data = spark.parallelize(vertices_with_index)
    # Perform clustering on data


def bind_coreset_construction(k: int, epsilon: float):
    def coreset_construction(input: Tuple[int, List[Vertex]]) -> Tuple[int, List[Vertex]]:

        # (Re)construct matrix of points
        points = np.array([[v.x, v.y] for v in input])
        weights = np.array([v.weight for v in input])

        n = len(points)
        
        # Get initial set of k centers
        c_points, c_indices = kmeans_plusplus(points, k)

        # Calculate the average distance between points and their closest center
        def dist_to_closest_center(p: np.ndarray) -> float:
            minimum = float("inf")
            for c in c_points:
                minimum = min(minimum, np.linalg.norm(p - c)**2)
            return minimum

        r_outer = 0.0
        r_inner = 0.0
        for p in points:
            r_outer = r_outer + dist_to_closest_center(p)
        r_outer = r_outer / n
    
        NUM_CELLS = 5
        grid_length = epsilon * r_outer / math.sqrt(2)
        cell_length = grid_length / NUM_CELLS

        def point_in_ball(p, c) -> bool:
            dist = abs(np.linalg.norm(p - c))
            return dist <= r_outer and dist > r_inner
        
        def point_in_grid(p, c) -> bool:
            return c[0] - grid_length/2 <= p[0] <= c[0] + grid_length/2 and c[1] - grid_length/2 <= p[1] <= c[1] + grid_length/2
        
        def point_cell(p, c) -> Tuple[int, int]:
            if not point_in_grid(p, c):
                return None

            col = (c[0] - p[0]) / cell_length
            row = (c[1] - p[1]) / cell_length
            
            return (col, row)
        
        def cell_center(c, col, row) -> np.ndarray:
            return np.array([c[0] - grid_length/2 + (col + 0.5) * cell_length, c[1] - grid_length/2 + (row + 0.5) * cell_length])
        
        # Store a list of dictionaries, one per ball/center point
        # The keys of each dictionary are cells of the balls and the values are
        # lists of indices in points that should be replaced by a single representative

        # Find the points that should be replaced by a single representative
        reps: List[Dict[Tuple[int, int], List[int]]] = [{}] * k
        for i in range(len(points)):
            p = points[i]
            for c in c_points:
                if not point_in_ball(p, c) or not point_in_grid(p, c):
                    continue
                cell = point_cell(p, c)
                reps[c][cell].append(i)
                
        # Replace the found points by their representative
        new_points = []

    
    return coreset_construction

def main() -> None:

    sparkConf = SparkConf().setAppName('AffinityClustering')
    spark = SparkContext(conf=sparkConf)
    
    data_X, data_y = create_blobs()

    spark.stop()

    # colors = ['#dc79c6' if i in c_indices else '#282a36'  for i in range(len(data_X))]

    fig, ax_data = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout()
    ax_data.set_title("Dataset")
    ax_data.scatter(data_X[:, 0], data_X[:, 1], marker="o", s=25)

    plt.show()

if __name__ == '__main__':
    # Initial call to main function
    main()