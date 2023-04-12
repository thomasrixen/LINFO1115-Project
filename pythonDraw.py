import networkx as nx
import matplotlib.pyplot as plt
from template_utils import *

def drawMyGraph(graph):

    # create an empty graph
    G = nx.Graph()

    # add edges to the graph
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    # plot the graph
    nx.draw(G, with_labels=True)
    plt.show()

drawMyGraph(network(pd.read_csv("CollegeMsg.csv")))