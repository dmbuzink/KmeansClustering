from typing import List, Tuple
from Vertex import Vertex
import random
import math
from pyspark import SparkConf, SparkContext, RDD, Broadcast, AccumulatorParam

# Store the SparkContext as a global variable
spark: SparkContext = None

def merge_coresets(coreset_a: List[Vertex], coreset_b: List[Vertex]):
    return coreset_a + coreset_b


def combine_into_pairs(coresets: List[List[Vertex]]):
    resultList = []
    for i in range(0, len(coresets), 2):
        resultList.append(merge_coresets(coresets[i], coresets[i + 1]))
    return resultList


def get_number_between_zero_and_one() -> float:
    number = 0
    while number == 0 or number == 1:
        number = random.random() 
    return number


def dist(v_a: Vertex, v_b: Vertex) -> float:
    return math.sqrt((v_a.x - v_b.x)**2 + (v_a.y - v_b.y)**2)


def get_centroids(points: List[Vertex], k: int) -> List[Vertex]:
    # See kMeans++
    centroids: List[Vertex] = []

    # Sample first centroid uniformly at random from input points
    x = get_number_between_zero_and_one()
    sampled_index = math.ceil(x * len(points))
    centroids.append(points[sampled_index])
    
    min_distances: List[float] = []
    for j in range(len(points)):
        min_distances.append(math.inf)
    
    for i in range(1, k):
        # Update distance to closest centroid for each input point
        for j in range(len(points)):
            x = dist(points[j], centroids[i - 1])
            if min_distances[j] > x:
                min_distances[j] = x

        cumulative: List[float] = []
        # Compute cumulative distribution of squared distances
        cumulative.append(min_distances[1]**2)
        for j in range(1, len(points)):
            cumulative.append(cumulative[j - 1] + (min_distances[j]**2))

        # Sample next centroid according to the distribution of squared distances
        x = get_number_between_zero_and_one() * cumulative[-1]

        # Determine index of sampled point
        if x <= cumulative[1]:
            sampled_index = 1
        else:
            for j in range(1, len(points)):
                if x > cumulative[j - 1] and x <= cumulative[j]:
                    sampled_index = j

        # Add sampled point to set of centroids
        centroids.append(points[sampled_index])
    
    return centroids


def construct_coreset(vertices: List[Vertex]) -> List[Vertex]:
    return vertices


def map_vertex_to_tuple(v: Vertex):
    return (v.index, v)


def cluster(points: List[Vertex]):
    """
    Takes a set of points and computes the coreset for these points.

    :param points: a list of point sets as a list of lists of vertices
    :return: a list of point sets as a list of lists of vertices [and possible some leader-esque information]
    """
    global spark

    # Initialize spark context
    sparkConf = SparkConf().setAppName('AffinityClustering')
    spark = SparkContext(conf=sparkConf)

    pointTuples = map(map_vertex_to_tuple, points)

    points_rdd = spark.parallelize(pointTuples)
    # points_rdd.

    


    return [] # set of representatives


# Main function
def main() -> None:
    # Datasets

    # Split all points over x sets
    # Coresets = generate coresets across m machines, with the points split across the machines roughly equally
    coresets = List[List[Vertex]]
    while len(coresets) > 1:
        newPointSets = combine_into_pairs(coresets)
        coresets = cluster(newPointSets)


if __name__ == '__main__':
    # Initial call to main function
    main()

