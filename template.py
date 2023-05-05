import pandas as pd
import math
import numpy as np
from template_utils import *
import matplotlib.pyplot as plt
import sys 
sys.setrecursionlimit(6000)

def Q1(dataframe):
    """
    Input: Pandas dataframe as described above representing a graph
    Output: [number_of_different_components, number_of_bridges, number_of_local_bridges]
    """
    print("Q1...")
    #Your code here
    schoolNetwork = SchoolNetwork(dataframe)
    graph = schoolNetwork.getNetwork()
    return [num_components(graph), len(bridges(graph)), count_local_bridges(graph)]

def Q2(dataframe):
    """
    Input: Pandas dataframe as described above representing a graph
    Output: Returns the total amount of triadic closures that arrise between the median timestamp of the dataframe until the last timestamp.
    """
    print("Q2...")
    schoolNetwork = SchoolNetwork(dataframe)
    return count_triadic_closures(schoolNetwork.getLastMessage())

def Q3(dataframe):
    """
    Input: Pandas dataframe as described above representing a graph
    Output: Returns a list where the element at index i represents the total number of small paths of distances i in the graph.
    Reminder: Take into account that the graph is directed now.
    """
    print("Q3...")
    schoolNetwork = SchoolNetwork(dataframe)
    graph = schoolNetwork.getDirectedNetwork()
    # Initialize the list of small paths with 0's
    small_paths_list = [0] * 10

    # Iterate over all nodes in the graph
    for node in graph:
        small_paths_list = bfs_small_paths(graph, node, small_paths_list)

    
    y = small_paths_list
    x = [i for i in range(len(y))]
    

    plt.plot(x, y, 'o-', label='Data')
    for i, j in zip(x, y):
        plt.annotate(str(j), xy=(i, j), ha='center', va='bottom')
    plt.xlabel('Number of Intermediaries')
    plt.ylabel('Number of Chains')
    plt.title('Graph Q3')
    plt.show()
    return small_paths_list



def Q4(dataframe):
    """
    Input: Pandas dataframe as described above representing a graph
    Output: (index, pagerank_value)
    where the index represents the id of the node with the highest pagerank value and pagerank_value its associated pagerank value after convergence.
    (we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-10))
    Reminder: Take into account that the graph is directed now.
    """
    print("Q4...")
    schoolNetwork = SchoolNetwork(dataframe)
    directedNetwork = schoolNetwork.getDirectedNetwork()
    pagerank_value = pagerank(directedNetwork)
    maxId = -1
    maxRank = -1
    for i in range(len(pagerank_value)):
        if (pagerank_value[i] > maxRank):
            maxRank = pagerank_value[i]
            maxId = i
            
    return (maxId, maxRank)
    
     # the id of the node with the highest pagerank score, the associated pagerank value.

    #Your code here
    
    #Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-10)

def Q5(dataframe):
    """
    Input: Pandas dataframe as described above representing a graph
    Output: the average local clustering coefficient
    """
    print("Q5...")
    schoolNetwork = SchoolNetwork(dataframe)
    graph = schoolNetwork.getNetwork()
    lcc = local_clustering_coefficient(graph)
    sum = 0
    for i in range(len(lcc)):
        sum += lcc[i]

    #Your code here
    return sum/len(lcc) #the average local clustering coefficient of the graph

#you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

df = pd.read_csv('CollegeMsg.csv')
# print(Q1(df))
# print(Q2(df))
# print(Q3(df))
print(Q4(df))
# print(Q5(df))