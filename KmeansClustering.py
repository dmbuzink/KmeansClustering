from typing import List
from Vertex import Vertex

def merge_coresets(coreset_a: List[Vertex], coreset_b: List[Vertex]):
    return coreset_a + coreset_b


def combine_into_pairs(coresets: List[List[Vertex]]):
    resultList = []
    for i in range(0, len(coresets), 2):
        resultList.append(merge_coresets(coresets[i], coresets[i + 1]))
    return resultList


def compute_coreset(pointSets: List[List[Vertex]]):
    """
    Takes a set of points and computes the coreset for these points.

    :param points: a list of point sets as a list of lists of vertices
    :return: a list of point sets as a list of lists of vertices [and possible some leader-esque information]
    """
    return [] # set of representatives


# Main function
def main() -> None:
    # Datasets

    # Split all points over x sets
    # Coresets = generate coresets across m machines, with the points split across the machines roughly equally
    coresets = List[List[Vertex]]
    while len(coresets) > 1:
        newPointSets = combine_into_pairs(coresets)
        coresets = compute_coreset(newPointSets)


if __name__ == '__main__':
    # Initial call to main function
    main()

