from typing import List, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs 
from sklearn.cluster import kmeans_plusplus

from pyspark import SparkConf, SparkContext, RDD

from Vertex import Vertex

spark: SparkContext = None

def create_blobs() -> Tuple[np.ndarray, np.ndarray]:
    n_samples = 100
    n_clusters = 3

    return make_blobs(n_samples=n_samples, random_state=8, centers=n_clusters)

def perform_clustering(P: np.ndarray, k: int, epsilon: float = 0.1, num_partitions: int = 10) -> np.ndarray:

    global spark
    
    # Create vertices for points
    vertices = [(i, Vertex(i, P[i][0], P[i][1], 1)) for i in range(len(P))]
    # Parallelize
    data = spark.parallelize(vertices)
    # Divide among machines
    B = len(vertices) / num_partitions
    data = data.map(lambda v: (math.floor(v[0] / B), v[1]))
    # Calculate coreset
    data = data.groupByKey().map(bind_coreset_construction(k, epsilon))
    # Flatten into list of vertices
    data_collect = data.reduce(lambda a, b: (min(a[0], b[0]), a[1] + b[1]))[1]

    # Merge and recalculate coreset...
    # ...

    # (Re)construct numpy matrix from vertices
    return np.array([[v.x, v.y] for v in data_collect])

def paoi(a, b):
    print(f"a: {a}")
    print(f"b: {a}")
    return (a[0], a[0] + b[0])

def bind_coreset_construction(k: int, epsilon: float):
    def coreset_construction(input: Tuple[int, List[Vertex]]) -> Tuple[int, List[Vertex]]:

        points = np.array([[v.x, v.y] for v in input[1]])
        weights = np.array([v.weight for v in input[1]])
        is_representative = [False] * len(points)

        n = len(points)
        
        # Get initial set of k centers
        c_points, c_indices = kmeans_plusplus(points, k)

        # Calculate the average distance between points and their closest center
        def dist_to_closest_center(p: np.ndarray) -> float:
            minimum = float("inf")
            for c in c_points:
                minimum = min(minimum, np.linalg.norm(p - c)**2)
            return minimum

        def replace_points(points, weights, is_representative, centers, r_outer, r_inner) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
            cell_length = epsilon * r_outer / math.sqrt(2)
            num_cells = math.ceil(r_outer / cell_length)
            grid_length = num_cells * cell_length

            def point_in_ball(p, c) -> bool:
                dist = abs(np.linalg.norm(p - c))
                return dist <= r_outer and dist > r_inner
            
            def point2cell(p, c) -> Tuple[int, int]:
                if not point_in_ball(p, c):
                    return None

                col = math.floor(abs(p[0] - (c[0] - grid_length/2)) / cell_length)
                row = math.floor(abs(p[1] - (c[1] - grid_length/2)) / cell_length)
                
                return (col, row)
            
            def cell_center(c, col, row) -> np.ndarray:
                ret = np.array([c[0] - grid_length/2 + (col + 0.5) * cell_length, c[1] - grid_length/2 + (row + 0.5) * cell_length])
                return ret

            # Initialize new list of points to return
            new_points: List[np.ndarray] = []
            new_weights: List[float] = []
            new_is_representative: List[bool] = []
            # For each ball, keep a dictionary that stores a list per cell containing
            # the weights of the points in that cell
            weight_of_cell: List[Dict[Tuple[int, int], float]] = [{}] * len(centers)

            # Iterate over all points and store them in the new_points list
            # or in the cell dictionary
            for i in range(len(points)):
                p = points[i]
                
                if is_representative[i]:
                    # Do nothing for representative points
                    new_points.append(p)
                    new_weights.append(weights[i])
                    new_is_representative.append(True)
                else:
                    # Check if this point falls in a ball
                    in_ball = False
                    for j in range(len(centers)):
                        c = centers[j]

                        if not point_in_ball(p, c):
                            continue

                        # The point falls in this ball
                        # Add the weight of the point to the cell
                        cell = point2cell(p, c)
                        if cell not in weight_of_cell[j]:
                            weight_of_cell[j][cell] = 0
                        weight_of_cell[j][cell] += weights[i]
                        in_ball = True
                        break
                    
                    if not in_ball:
                        # The point is not inside a ball, just keep it
                        new_points.append(p)
                        new_weights.append(weights[i])
                        new_is_representative.append(is_representative[i])
            
            # Insert a new point per cell in the dictionary,
            # with weight equal to the summed weights of the points in the cell
            for j in range(len(centers)):
                c = centers[j]
                # Iterate over all cells and add point per cell
                for cell in weight_of_cell[j]:
                    cell_c = cell_center(c, cell[0], cell[1])
                    weight = weight_of_cell[j][cell]
                    
                    new_points.append(cell_c)
                    new_weights.append(weight)
                    new_is_representative.append(True)

            return np.array(new_points), np.array(new_weights), new_is_representative
        
        # Calculate the average distance r
        r = 0.0
        for p in points:
            r = r + dist_to_closest_center(p)
        r = r / n

        r_outer = r
        r_inner = 0.0

        # print(f"Count: {is_representative.count(False)}")
        # Replace points inside balls with a weighted representative point
        points, weights, is_representative = replace_points(points, weights, is_representative, c_points, r_outer, r_inner)
        # print(f"Count: {is_representative.count(False)}")
        # Repeat this process until all points are replaced by a rep
        # j = 1
        # while False in is_representative:
        #     r_inner = r_outer
        #     r_outer = 2**j * r
        #     points, weights, is_representative = replace_points(points, weights, is_representative, c_points, r_outer, r_inner)
        #     # print(f"Count: {is_representative.count(False)}")

        # Return coreset
        vertices: List[Vertex] = []
        for i in range(len(points)):
            p = points[i]
            vertices.append(Vertex(i, p[0], p[1], weights[i]))

        return (input[0], vertices)

    
    return coreset_construction

def main() -> None:

    global spark

    sparkConf = SparkConf().setAppName('AffinityClustering')
    spark = SparkContext(conf=sparkConf)
    
    data_X, data_y = create_blobs()
    clustered = perform_clustering(data_X, 3)

    spark.stop()

    # colors = ['#dc79c6' if i in c_indices else '#282a36'  for i in range(len(data_X))]

    fig, ax_data = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout()
    ax_data.set_title("Dataset")
    ax_data.scatter(clustered[:, 0], clustered[:, 1], marker="o", s=25)

    plt.show()

if __name__ == '__main__':
    # Initial call to main function
    main()