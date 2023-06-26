import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, Optional

class Graph():
    def __init__(self, num_vertices: int = 0) -> None:
        """Initializes a Graph instance.

        Args:
            num_vertices: number of vertices.
        """

        if num_vertices < 0:
            message = f"num_vertices must be greater than 0; got {num_vertices}"
            raise ValueError(message)

        self.num_vertices = num_vertices
        self.edges = self._init_edges()

        # used in visualization
        self._num_iterations = 50
        self._threshold = 0.01
        self._eps = 0.001

    def _init_edges(self) -> Dict:
        """Initializes an edge dict with no edges.

            Returns:
                Dict: an edge dict without edges.
        """

        return {v: set() for v in range(self.num_vertices)}


    def add_vertex(self) -> None:
        """Adds a vertex to the graph.
        """

        self.num_vertices += 1
        self.edges[self.num_vertices - 1] = set()

    def add_edge(self, v: int, u: int) -> None:
        """Adds an edge for 2 given vertices. Vertices must be
            distinct and in the range [0, self.num_vertices).

        Args: 
            v: vertex id.
            u: vertex id.
        """

        if v >= self.num_vertices:
            message = f"v must be less than self.num_vertices, or {self.num_vertices}; got {v}."
            raise ValueError(message)
        if u >= self.num_vertices:
            message = f"u must be less than self.num_vertices, or {self.num_vertices}; got {u}."
            raise ValueError(message)

        self.edges[v].add(u)
        self.edges[u].add(v)

    def make_random(self, p: float) -> None:
        """Creates a random graph with the current number of vertices.

        Args:
            p: probability two given vertices are connected by an edge.
        """

        if p < 0 or p > 1:
            message = f"p must be in the range between 0 and 1; got {p}."
            raise ValueError(message)

        num_pairs = int(self.num_vertices * (self.num_vertices + 1) / 2)
        edge_exists = np.random.binomial(n=1, p=p, size=num_pairs)
        
        self.edges = self._init_edges()
        for i, v in enumerate(range(self.num_vertices)):
            for j, u in enumerate(range(0, i)):
                if edge_exists[int(i * (i + 1) / 2 + j)]:
                    self.add_edge(v, u)
            

    def draw(self, width: float, height: float, fpath: Optional[str] = None) -> None:
        """Visualizes the graph with matplotlib using a spring layout.

        Args:
            width: width in plot figure size.
            height: height in plot figure size.
            num_iterations: maximum number of iterations in layout calculation.
            threshold: threshold of tolerance to stop layout calculation.
        """

        pos = self._get_layout(width, height)

        plt.figure(figsize=(width, height))
        plt.scatter(pos[:,0], pos[:,1])

        for v in range(self.num_vertices):
            for u in self.edges[v]:
                if v > u:
                    plt.plot([pos[v][0], pos[u][0]], [pos[v][1], pos[u][1]])

        for v in range(self.num_vertices):
            plt.annotate(v, pos[v])

        plt.axis("off")

        if fpath:
            plt.savefig(fpath)
        else:
            plt.show()

    def _get_layout(self, width: float, height: float) -> np.ndarray:
        """Calculates a spring layout.

        Args:
            width: width in plot figure size.
            height: height in plot figure size.
            num_iterations: maximum number of iterations in layout calculation.
            threshold: threshold of tolerance to stop layout calculation.
            eps: epsilon to avoid division by zero.

        Returns:
            np.ndarray: numpy array of 2d vertex positions in layout.
        """

        area = width * height
        k = np.sqrt(area / self.num_vertices)

        pos = np.random.uniform((0, 0), (1, 1), size=(self.num_vertices, 2))
        disp = np.zeros((self.num_vertices, 2))

        def f_a(z):
            return np.linalg.norm(z) / k
        def f_r(z):
            return k ** 2 / (np.linalg.norm(z) + self._eps)
        
        t = 1.0
        dt = t / (self._num_iterations + 1)

        def cool(t):
            return t - dt

        for _ in range(self._num_iterations):
            # calculate repulsive forces
            for v in range(self.num_vertices):
                disp[v] = [0, 0]
                for u in range(self.num_vertices):
                    if u not in self.edges[v]:
                        delta = pos[v] - pos[u] + self._eps
                        disp[v] += delta / np.linalg.norm(delta) * f_r(delta)
            
            # calculate attractive forces
            for v in range(self.num_vertices):
                for u in self.edges[v]:
                    delta = pos[v] - pos[u] + self._eps
                    disp[v] -= delta / np.linalg.norm(delta) * f_a(delta)
                    disp[u] += delta / np.linalg.norm(delta) * f_a(delta)
            
            # update positions
            delta_pos = np.zeros((self.num_vertices, 2))
            for v in range(self.num_vertices):
                pos_v = pos[v] + disp[v] / (np.linalg.norm(disp[v]) + self._eps) * t
                delta_pos[v] = pos_v - pos[v]
                pos[v] = pos_v

            # check significance of updates
            if np.linalg.norm(delta_pos) / self.num_vertices < self._threshold:
                break

            t = cool(t)

        return pos
