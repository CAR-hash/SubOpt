import math

import cost_manager
from base_task import BaseTask
import random
from typing import Set, List
import numpy as np
import pickle
from collections import defaultdict
import os

import networkx as nx
from scipy.io import mmread

class RevenueMax(BaseTask):
    """
    Select a set of users in a social network to advertise a product to maximize the revenue.

    Model social network as undirected graph.

    The objective is a non-monotone submodular function.
    """

    def __init__(self, budget: float, m: int = None, edges: List[List[int]] = None, pckl_path: str = None, seed=1):
        """
        Inputs:
        - m: number of nodes
        """
        super().__init__()
        random.seed(seed)
        if pckl_path is None:
            if m is None or edges is None:
                raise ValueError("Missing Parameters.")
            self.num_nodes = m
            self.num_edges = len(edges)
        else:
            if not os.path.exists(pckl_path):
                raise OSError("Path is not valid.")
            with open(pckl_path, "rb") as rd:
                params = pickle.load(rd)
            self.num_nodes = params['num_vertices']
            edges = params['edges']
            self.num_edges = len(params['edges'])

        self.objects = [i for i in range(self.num_nodes)]
        self.weights = np.zeros(shape=(self.num_nodes, self.num_nodes))
        self.adjacent_table = [[] for _ in range(self.num_nodes)]
        self.adjacency = defaultdict(set)
        for u, v in edges:
            w = random.random()
            self.adjacent_table[u].append((v, w))
            self.adjacent_table[v].append((u, w))
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)
            self.weights[u][v] = w
            self.weights[v][u] = w

        self.costs_obj = [self._compute_cost(u) for u in self.objects]

        self.b = budget

        # self.objective_dict = {'internal': self.internal_objective}

    @property
    def ground_set(self):
        return self.objects

    def _compute_cost(self, u):
        t = 0.
        mu = 0.2
        for v, w in self.adjacent_table[u]:
            t += w
        return 1 - np.exp(- mu * np.sqrt(t))

    def internal_objective(self, S):
        S_set = set(S)
        N_set = set(self.ground_set)
        S_complement = N_set - S_set
        res = 0.

        for v in S_complement:
            partial_sum = 0.
            for u in S:
                if u in self.adjacency[v]:
                    partial_sum += self.weights[u][v]
            res += np.sqrt(partial_sum)
        return res

    def cost_of_set(self, S: List[int]):
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        assert singleton < len(
            self.costs_obj), "Singleton: {}".format(singleton)
        return self.costs_obj[singleton]


class CalTechMaximization(BaseTask):
    def __init__(self, budget: float, n: int = None, graph_path: str = None, knapsack=True, seed = 0,
                 prepare_max_pair=True, print_curvature=False, construct_graph = False, min_cost = 0.4, factor = 4.0, cost_mode = "normal", graph_suffix = "", enable_packing = False, constraint_count = 4):
        """
        Inputs:
        - n: max_nodes
        - b: budget
        """
        super().__init__()
        if graph_path is None:
            raise Exception("Please provide a graph.")
        np.random.seed(seed)
        random.seed(seed)

        self.enable_packing_constraint = enable_packing
        self.cc = constraint_count

        self.max_nodes = n
        self.graph_path = graph_path

        self.costs_obj = []
        self.weights = {}

        graph_name = "graph" + graph_suffix + ".txt"
        cost_name = "costs" + graph_suffix + ".txt"

        # cost parameters
        if construct_graph:
            self.graph: nx.Graph = self.load_original_graph(graph_path + "/" + "graph.mtx")

            self.nodes = list(self.graph.nodes)
            self.nodes = [int(node_str) for node_str in self.nodes]
            self.nodes.sort()
            self.objs = list(range(0, len(self.nodes)))

            with open(self.graph_path + "/" + "weights" + graph_suffix + ".txt", "w") as f:
                for edge in self.graph.edges:
                    w = random.random()
                    # f.write(f"{edge[0]} {edge[1]} {w}\n")
                    self.weights[(int(edge[0]), int(edge[1]))] = w
                    self.weights[(int(edge[1]), int(edge[0]))] = w

            # nx.write_adjlist(G=self.graph, path=self.graph_path + "/" + graph_name)

            if cost_mode == 'normal':
                self.assign_costs(knapsack, cost_mode)
            else:
                cm = cost_manager.CostManager()
                cm.set_model(self)
                cm.set_mode(cost_mode)
                cm.build()

                self.costs_obj = cm.assign()

            # with open(self.graph_path + "/" + cost_name, "w") as f:
            #     for node in range(0, self.max_nodes):
            #         f.write(f"{self.costs_obj[node]}\n")



        else:
            self.graph: nx.Graph = self.load_graph(graph_path + "/" + graph_name)

            self.nodes = list(self.graph.nodes)
            self.nodes = [int(node_str) for node_str in self.nodes]
            self.nodes.sort()
            self.objs = list(range(0, len(self.nodes)))

            with open(self.graph_path + "/" + cost_name, "r") as f:
                while True:
                    line = f.readline()
                    if line == "":
                        break
                    self.costs_obj.append(float(line.rstrip("\n")))

            with open(self.graph_path + "/" + "weights" + graph_suffix + ".txt", "r") as f:
                while True:
                    line = f.readline()
                    if line == "":
                        break
                    args = line.rstrip("\n").split(" ")
                    start = int(args[0])
                    stop = int(args[1])
                    weight = float(args[2])
                    self.weights[(start, stop)] = weight
                    self.weights[(stop, start)] = weight

        self.budget = budget

        if not knapsack:
            self.costs_obj = [
                1
                for _ in self.objs
            ]

        if prepare_max_pair:
            self.prepare_max_2_pair()

        if print_curvature:
            self.print_curvature()

    @property
    def ground_set(self):
        return self.objs

    def load_original_graph(self, path: str):
        if not os.path.isfile(path):
            raise OSError("File *.txt does not exist.")

        sparse = mmread(path)

        intact_graph: nx.Graph = nx.Graph(sparse)

        # print(self.max_nodes)

        if self.max_nodes <= len(intact_graph.nodes):
            nodes = random.sample(list(intact_graph.nodes), self.max_nodes)
            # print(len(nodes))
            return intact_graph.subgraph(nodes)
        else:
            return intact_graph

    def load_graph(self, path: str):
        if not os.path.isfile(path):
            raise OSError("File *.txt does not exist.")
        intact_graph: nx.Graph = nx.read_adjlist(path)

        return intact_graph


    def internal_objective(self, S: List[int]):
        """
        Inputs:
        - S: solution set
        - llambda: coefficient which lies in [0,1]
        """
        if type(S) == int:
            S = [S]

        ret = 0

        S = list(S)

        for x in self.nodes:
            w = 0.

            neighbors = set(self.graph.neighbors(x))
            neighbors = set([int(n) for n in neighbors])

            for v in S:
                nv = self.nodes[v]
                if nv in neighbors:
                    w += self.weights[(x, nv)]

            ret += math.pow(w, 0.9)

        return ret

    def cost_of_set(self, S: List[int]):
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        assert singleton < len(
            self.costs_obj), "Singleton: {}".format(singleton)
        return self.costs_obj[singleton]


def main():
    model = RevenueMax(budget=1.0, pckl_path="dataset/revenue/25_youtube_top5000.pkl")

    S = [0, 1, 2, 3, 4]
    print("S =", S)
    print("f(S) =", model.objective(S))
    print("c(S) =", model.cost_of_set(S))


if __name__ == "__main__":
    main()
