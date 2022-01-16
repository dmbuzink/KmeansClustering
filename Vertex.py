from typing import List
import numpy as np

class Vertex:
    """
    A weighed point in 2D space.
    """

    def __init__(self, index: int, x: float, y: float, z: float, weight: float = 1) -> 'Vertex':
        """
        Creates a `Vertex`.

        Parameters
        ----------

        index : The index of the vertex in the graph.

        x : The x-coordinate of the vertex.

        y : The y-coordinate of the vertex.

        w : The weight of the representative

        Returns
        -------

        The created vertex.
        """

        self.index: int = index
        self.position: np.ndarray = np.array([x, y, z])
        self.weight: float = weight
        self.is_representative: bool = False
        self.representing: List[Vertex] = [self]

    def equals(self, other: 'Vertex') -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z

    def copy(self) -> 'Vertex':
        return Vertex(self.index, self.x, self.y, self.z)
    
    def __str__(self) -> str:
        return f"[{self.index}]: ({self.position})"
    
    def __repr__(self) -> str:
        return self.__str__(self)