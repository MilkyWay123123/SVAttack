import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

from .tools import *

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = normalize_adjacency_matrix(self.A_binary)


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # graph = AdjMatrixGraph()
    # A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    # f, ax = plt.subplots(1, 3)
    # ax[0].imshow(A_binary_with_I, cmap='gray')
    # ax[1].imshow(A_binary, cmap='gray')
    # ax[2].imshow(A, cmap='gray')
    # plt.show()
    #print(A_binary_with_I.shape, A_binary.shape, A.shape)
    print(Graph().A.shape)