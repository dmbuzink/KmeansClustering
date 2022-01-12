class Vertex:
    """
    A weighed point in 2D space.
    """

    def __init__(self, index: int, x: float, y: float, weight: float = 1) -> 'Vertex':
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
        self.x: float = x
        self.y: float = y
        self.weight: float = weight

    def equals(self, other: 'Vertex') -> bool:
        return self.x == other.x and self.y == other.y

    def copy(self) -> 'Vertex':
        return Vertex(self.index, self.x, self.y)