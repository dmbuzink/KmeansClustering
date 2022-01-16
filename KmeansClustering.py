from typing import Iterable, List, Tuple, Dict
from functools import reduce
import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs 
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import KMeans

from pyspark import SparkConf, SparkContext, RDD

from Vertex import Vertex

spark: SparkContext = None

fig: plt.Figure = None
ax_data: plt.Axes = None
ax_coreset: plt.Axes = None

def create_blobs() -> Tuple[np.ndarray, np.ndarray]:
    n_samples = 1000
    n_clusters = 3

    return make_blobs(n_samples=n_samples, random_state=8, centers=n_clusters)

def perform_clustering(P: np.ndarray, k: int, epsilon: float = 0.1, num_partitions: int = 16) -> np.ndarray:

    global spark
    # global ax_coreset

    # attempt at adding the pyspark stuff
    def merge_coreset_keys(key: int) -> int:
        return math.floor(key / 2)

    def merge_lists(llist: Tuple[int, List[List[Vertex]]]) -> Tuple[int, List[Vertex]]:
        l = [item for sublist in llist[1] for item in sublist]
        return (llist[0], l)

    def mark_not_repr(tup: Tuple[int, List[Vertex]]) -> Tuple[int, List[Vertex]]:
        for v in tup[1]:
            v.is_representative = False
        return tup

    vertices = [(i, Vertex(i, P[i][0], P[i][1], P[i][2], 1)) for i in range(len(P))]
    # Parallelize
    B = len(vertices) / num_partitions
    data = spark.parallelize(vertices)
    data = data.map(lambda v: (math.floor(v[0] / B), v[1]))
    data = data.groupByKey()

    
    data = data.map(bind_coreset_construction(k, epsilon))
    
    while data.count() > 1:
        print(data.count())
        # Merge coresets
        data = data.map(lambda v: (merge_coreset_keys(v[0]), v[1]))
        data = data.groupByKey().map(merge_lists).map(mark_not_repr)

        # Calculate coresets
        data = data.map(bind_coreset_construction(k, epsilon))

        
    # Flatten into list of vertices
    result = data.reduce(lambda a, b: (min(a[0], b[0]), a[1] + b[1]))[1]

    # Perform K-means clustering on result
    # Reconstruct numpy matrix
    points = np.array([v.position for v in result])
    # return points
    weights = np.array([v.weight for v in result])

    kmeans = KMeans(n_clusters=min(len(points), k), random_state=0).fit(points, sample_weight=weights)

    # ax_coreset.scatter(points[:, 0], points[:, 1], marker="o", c=kmeans.labels_, s=25)

    # Build index/class list for all points using the index/class of the representative points
    result_values = [-1] * len(vertices)
    labels = [-1] * len(vertices)
    for i in range(len(result)):
        v = result[i]
        for r in v.representing:
            labels[r.index] = kmeans.labels_[i]
            result_values[r.index] = v.position

    if -1 in labels:
        print("Something went wrong with the labels")

    # return np.array(labels)
    return np.array(result_values)


def kmeans_local(P: np.ndarray, k: int, epsilon: float = 0.01, num_partitions: int = 10) -> np.ndarray:

    global ax_coreset

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
        
        if len(data) == 1:
            break
        # We want the vertices to behave as non-representative points again
        for t in data:
            for v in t[1]:
                v.is_representative = False
        # Recompute coreset
        data = list(map(bind_coreset_construction(k, epsilon), data))

    result = data[0][1]
    print(f"Result points: {len(result)}")

    # Perform K-means clustering on result
    # Reconstruct numpy matrix
    points = np.array([v.position for v in result])
    weights = np.array([v.weight for v in result])

    kmeans = KMeans(n_clusters=k, random_state=0).fit(points, sample_weight=weights)

    ax_coreset.scatter(points[:, 0], points[:, 1], marker="o", c=kmeans.labels_, s=25)

    # Build index/class list for all points using the index/class of the representative points
    labels = [-1] * len(vertices)
    for i in range(len(result)):
        v = result[i]
        for r in v.representing:
            labels[r.index] = kmeans.labels_[i]

    if -1 in labels:
        print("Something went wrong with the labels")

    return np.array(labels)

def bind_coreset_construction(k: int, epsilon: float):
    def coreset_construction(input: Tuple[int, List[Vertex]]) -> Tuple[int, List[Vertex]]:

        vertices = input[1]
        n = len(vertices)

        print(f"Start: {n}")

        points = np.array([v.position for v in vertices])
        
        # Get initial set of k centers
        c_points, c_indices = kmeans_plusplus(points, k)

        # Calculate the average distance between points and their closest center
        def dist_to_closest_center(p: np.ndarray) -> float:
            minimum = float("inf")
            for c in c_points:
                minimum = min(minimum, np.linalg.norm(p - c)**2)
            return minimum

        def replace_points(vertices: List[Vertex], centers: np.ndarray, r_outer: float, r_inner: float) -> List[Vertex]:

            vertices = list(vertices)

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

            def point3cell(p, c) -> Tuple[int, int, int]:
                if not point_in_ball(p, c):
                    return None

                col = math.floor(abs(p[0] - (c[0] - grid_length/2)) / cell_length)
                row = math.floor(abs(p[1] - (c[1] - grid_length/2)) / cell_length)
                depth = math.floor(abs(p[2] - (c[2] - grid_length/2)) / cell_length)

                return (col, row, depth)
            
            def cell_center(c, col, row) -> np.ndarray:
                ret = np.array([c[0] - grid_length/2 + (col + 0.5) * cell_length, c[1] - grid_length/2 + (row + 0.5) * cell_length])
                return ret

            def cell_center3(c, col, row, depth) -> np.ndarray:
                ret = np.array([c[0] - grid_length/2 + (col + 0.5) * cell_length, c[1] - grid_length/2 + (row + 0.5) * cell_length, c[2] - grid_length/2 + (depth + 0.5) * cell_length])
                return ret

            # Initialize new list of points to return
            # For each ball, keep a dictionary that stores a list per cell containing
            # the points in that cell
            new_vertices: List[Vertex] = []
            points_in_cell: Dict[int, Dict[Tuple[int, int], List[Vertex]]] = {i: {} for i in range(len(centers))}

            # Iterate over all points and store them in the new_points list
            # or in the cell dictionary
            for i in range(len(vertices)):
                v = vertices[i]
                
                if v.is_representative:
                    # Do nothing for representative points
                    new_vertices.append(v)
                else:
                    # Check if this point falls in a ball
                    in_ball = False
                    for j in range(len(centers)):
                        c = centers[j]

                        if not point_in_ball(v.position, c):
                            continue

                        # The point falls in this ball
                        # cell = point2cell(v.position, c)
                        cell = point3cell(v.position, c)

                        if cell not in points_in_cell[j]:
                            points_in_cell[j][cell] = []
                        points_in_cell[j][cell].append(v)

                        in_ball = True
                        break
                    
                    if not in_ball:
                        # The point is not inside a ball, just keep it
                        new_vertices.append(v)
            
            # Insert a new point per cell in the dictionary,
            # with weight equal to the summed weights of the points in the cell
            for j in range(len(centers)):
                c = centers[j]
                # Iterate over all cells and add point per cell
                for cell in points_in_cell[j]:
                    # cell_c = cell_center(c, cell[0], cell[1])
                    cell_c = cell_center3(c, cell[0], cell[1], cell[2])
                    in_cell = points_in_cell[j][cell]

                    representing: List[Vertex] = []
                    for v in in_cell:
                        representing += v.representing

                    weight = sum(v.weight for v in in_cell)
                    index = min(v.index for v in representing)
                    
                    v = Vertex(index, cell_c[0], cell_c[1], cell_c[2], weight)
                    v.is_representative = True
                    v.representing = representing

                    new_vertices.append(v)

            return new_vertices
        
        # Calculate the average distance r
        r = 0.0
        for p in points:
            r = r + dist_to_closest_center(p)
        r = r / n

        r_outer = r
        r_inner = 0.0

        # Replace points inside balls with a weighted representative point
        vertices = replace_points(vertices, c_points, r_outer, r_inner)

        j = 1
        while False in [v.is_representative for v in vertices]:
            r_inner = r_outer
            r_outer = 2**j * r
            vertices = replace_points(vertices, c_points, r_outer, r_inner)
            j += 1
        
        print(f"Subset {input[0]} points: {len(vertices)}")

        return (input[0], vertices)

    
    return coreset_construction

def main() -> None:

    global spark
    global fig
    global ax_data
    global ax_coreset

    sparkConf = SparkConf().setAppName('AffinityClustering')
    spark = SparkContext(conf=sparkConf)

    fig, (ax_coreset, ax_data) = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout()
    ax_data.set_title("Data")
    ax_coreset.set_title("Coreset")
    
    data_X, data_y = create_blobs()

    # labels = kmeans_local(data_X, 3)
    labels = perform_clustering(data_X, 3)

    spark.stop()

    ax_data.scatter(data_X[:, 0], data_X[:, 1], marker="o", c=labels, s=25)

    plt.show()


def get_kmeans_clustering(data, k):
    global spark

    sparkConf = SparkConf().setAppName('AffinityClustering')
    spark = SparkContext(conf=sparkConf)

    clustering = perform_clustering(data, k, epsilon=0.001)

    spark.stop()

    return clustering


if __name__ == '__main__':
    # Initial call to main function
    main()