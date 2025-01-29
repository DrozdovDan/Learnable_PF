import numpy.typing as npt
import typing as tp


class Map:
    def __init__(self, cells: npt.NDArray):
        self._width = cells.shape[1]
        self._height = cells.shape[0]
        self._cells = cells

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= j < self._width and 0 <= i < self._height

    def traversable(self, i: int, j: int) -> bool:
        return not self._cells[i, j]

    def get_neighbors(self, i: int, j: int) -> tp.List[tp.Tuple[int, int]]:
        neighbors = []
        delta = ((0, 1), (1, 0), (0, -1), (-1, 0))
        for dx, dy in delta:
            ni, nj = i + dx, j + dy
            if self.in_bounds(ni, nj) and self.traversable(ni, nj):
                neighbors.append((ni, nj))
        
        delta = ((1, 1), (1, -1), (-1, 1), (-1, -1))
        for dx, dy in delta:
            ni, nj = i + dx, j + dy
            if self.in_bounds(ni, nj) and self.traversable(ni, nj) \
                and self.traversable(ni, j) and self.traversable(i, nj):
                neighbors.append((ni, nj))

        return neighbors

    def get_size(self) -> tp.Tuple[int, int]:
        return self._height, self._width


class Node:
    def __init__(
        self,
        i: int,
        j: int,
        g: tp.Union[float, int] = 0,
        h: tp.Union[float, int] = 0,
        f: tp.Optional[tp.Union[float, int]] = None,
        parent: "Node" = None,
    ):
        
        self.i = i
        self.j = j
        self.g = g
        self.h = h
        if f is None:
            self.f = self.g + h
        else:
            self.f = f
        self.parent = parent

    def __eq__(self, other):
        
        return self.i == other.i and self.j == other.j

    def __lt__(self, other):
        
        return self.f < other.f

    def __hash__(self):
        return (self.i, self.j).__hash__()
    
    def __str__(self):
        return f"{self.i}, {self.j}, {self.h}, {self.g}, {self.f}"
