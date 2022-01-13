from typing import List, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs 
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import KMeans

from pyspark import SparkConf, SparkContext, RDD

from Vertex import Vertex

spark: SparkContext = None

def create_blobs() -> Tuple[np.ndarray, np.ndarray]:
    n_samples = 100000
    n_clusters = 3

    return make_blobs(n_samples=n_samples, random_state=8, centers=n_clusters)

def perform_clustering(P: np.ndarray, k: int, epsilon: float = 0.1, num_partitions: int = 10) -> np.ndarray:

    pass
    # global spark
    
    # # Create vertices for points
    # vertices = [(i, Vertex(i, P[i][0], P[i][1], 1)) for i in range(len(P))]
    # # Parallelize
    # data = spark.parallelize(vertices)
    # # Divide among machines
    # B = len(vertices) / num_partitions
    # data = data.map(lambda v: (math.floor(v[0] / B), v[1]))
    # # Calculate coreset
    # data = data.groupByKey().map(bind_coreset_construction(k, epsilon))
    # # Flatten into list of vertices
    # data_collect = data.reduce(lambda a, b: (min(a[0], b[0]), a[1] + b[1]))[1]

    # # Merge and recalculate coreset...
    # # ...

    # # (Re)construct numpy matrix from vertices
    # return np.array([[v.x, v.y] for v in data_collect])


def kmeans_local(P: np.ndarray, k: int, epsilon: float = 0.1, num_partitions: int = 10) -> Tuple[np.ndarray, np.ndarray]:

    # Create list of vertices
    vertices = [Vertex(i, P[i][0], P[i][1], 1) for i in range(len(P))]
    # Define bucket size
    B = len(vertices) / num_partitions
    # Divide vertices among buckets
    vertices_dict: Dict[int, List[Vertex]] = {i: [] for i in range(num_partitions)}
    for v in vertices:
        key = math.floor(v.index / B)
        vertices_dict[key].append(v)
    data = list(vertices_dict.items())

    # Compute coreset
    data = list(map(bind_coreset_construction(k, epsilon), data))

    # Combine and recompute while we have more than one group
    while len(data) > 1:
        # Combine pairs
        i = 0
        while i < len(data) - 1:
            data[i] = (data[i][0], data[i][1] + data[i + 1][1])
            data.pop(i + 1)
            i += 1
        # Recompute coreset
        if len(data) == 1:
            break
        data = list(map(bind_coreset_construction(k, epsilon), data))

    result = data[0][1]
    print(f"Result points: {len(result)}")

    # Perform K-means clustering on result
    # Reconstruct numpy matrix
    points = np.array([[v.x, v.y] for v in result])
    weights = np.array([v.weight for v in result])

    kmeans = KMeans(n_clusters=k, random_state=0).fit(points, sample_weight=weights)

    return np.array([[v.x, v.y] for v in result]), kmeans.labels_

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
                return dist <= r_outer and (dist > r_inner or r_inner == 0.0)
            
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
            weight_of_cell: Dict[int, Dict[Tuple[int, int], float]] = {i: {} for i in range(len(centers))}

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
            
            np_points = np.array(new_points)

            return np_points, np.array(new_weights), new_is_representative
        
        # Calculate the average distance r
        r = 0.0
        for p in points:
            r = r + dist_to_closest_center(p)
        r = r / n

        r_outer = r
        r_inner = 0.0

        # Replace points inside balls with a weighted representative point
        points, weights, is_representative = replace_points(points, weights, is_representative, c_points, r_outer, r_inner)

        j = 1
        while False in is_representative:
            r_inner = r_outer
            r_outer = 2**j * r
            points, weights, is_representative = replace_points(points, weights, is_representative, c_points, r_outer, r_inner)
            j += 1

        # Return coreset
        vertices: List[Vertex] = []
        for i in range(len(points)):
            p = points[i]
            vertices.append(Vertex(i, p[0], p[1], weights[i]))

        
        print(f"Subset {input[0]} points: {len(vertices)}")

        return (input[0], vertices)

    
    return coreset_construction

def main() -> None:

    global spark

    # sparkConf = SparkConf().setAppName('AffinityClustering')
    # spark = SparkContext(conf=sparkConf)
    
    data_X, data_y = create_blobs()

    c_points, c_indices = kmeans_local(data_X, 3)

    # spark.stop()

    fig, ax_data = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout()
    ax_data.set_title("Dataset")
    ax_data.scatter(c_points[:, 0], c_points[:, 1], marker="o", c=c_indices, s=25)

    plt.show()

if __name__ == '__main__':
    # Initial call to main function
    main()