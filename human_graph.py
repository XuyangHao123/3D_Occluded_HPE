import numpy as np


def get_graph():
    parent = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    human_graph = np.zeros([17, 17])
    for i, j in enumerate(parent):
        human_graph[i+1, j] = 1
        human_graph[j, i+1] = 1
    return human_graph
