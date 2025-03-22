from typing import Tuple
import numpy as np
from collections import deque

def phi_1(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def is_connected(N: np.ndarray) -> bool:
    used = np.zeros(N.shape[0])
    q = deque([0])
    used[0] = 1
    while q:
        v = q.popleft()
        for u in range(N.shape[0]):
            if N[v, u] > 0 and not used[u]:
                used[u] = 1
                q.append(u)
    return np.all(used)
class GaoData:
    """
    p: number of players
    edge_prob: probability of edge between two players
    L: number of games between two players
    v_range: range of v

    generates connected Erdos-Renyi graph with edge_prob probability ! bias, it's connected?
    generates v_i ~ uniform(v_range)
    """
    def __init__(
        self, p: int, edge_prob: float, L: int, v_range: Tuple[float, float]
    ):
        self.p = p
        self.edge_prob = edge_prob
        self.v_range = v_range
        self.L = L
        self.F = np.zeros((p, p))
        self.v = np.zeros(p)
        self.N = np.zeros((p, p))
        self.D = np.zeros(p)
        self.S = np.zeros((p, p))
        self.generate_v()
        self.generate_graph()
        self.generate_games()

    def generate_v(self):
        self.v = (
            np.random.rand(self.p) * (self.v_range[1] - self.v_range[0])
            + self.v_range[0]
        )

    def generate_graph(self):
        for i in range(self.p):
            for j in range(i + 1, self.p):
                if np.random.rand() < self.edge_prob:
                    self.N[i, j] = self.N[j, i] = self.L
        if not is_connected(self.N):
            self.N = np.zeros((self.p, self.p))
            self.generate_graph()

    def generate_games(self):
        for i in range(self.p):
            for j in range(i + 1, self.p):
                if self.N[i, j] > 0:
                    self.S[i, j] = np.random.binomial(
                        n=self.N[i, j], p=phi_1(self.v[i] - self.v[j])
                    )
                    self.S[j, i] = self.N[i, j] - self.S[i, j]