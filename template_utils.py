#if needed, write here your additional fuctions/classes with their signature and use them in the exercices:
# a specific place is available to copy them at the end of the Inginious task.

#First, import the libraries needed for your helper functions
import networkx as nx  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def num_components(graph):
    """Compute the number of components in a graph."""
    visited = set()
    num_components = 0
    for node in graph:
        if node not in visited:
            num_components += 1
            dfs(graph, node, visited)
    return num_components


def bridges(graph):
    """Compute the bridges in a graph."""
    visited = set()
    bridges = set()
    parent = {}
    low = {}
    for node in graph:
        if node not in visited:
            dfs_bridge(graph, node, visited, parent, low, bridges)
    return bridges


def dfs(graph, node, visited):
    """Depth-first search traversal of a graph."""
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)


def dfs_bridge(graph, node, visited, parent, low, bridges):
    """Depth-first search traversal of a graph to find bridges."""
    visited.add(node)
    low[node] = node
    for neighbor in graph[node]:
        if neighbor not in visited:
            parent[neighbor] = node
            dfs_bridge(graph, neighbor, visited, parent, low, bridges)
            low[node] = min(low[node], low[neighbor])
            if low[neighbor] == neighbor:
                bridges.add((node, neighbor))
        elif neighbor != parent.get(node):
            low[node] = min(low[node], neighbor)

def local_bridge(graph):
    """Compute all the local bridges in a graph."""
    visited = set()
    bridges = set()
    parent = {}
    low = {}
    dist = {}
    
    def dfs_bridge(node, depth):
        """Depth-first search traversal of a graph to find bridges and compute distance from root."""
        visited.add(node)
        low[node] = node
        dist[node] = depth
        for neighbor in graph[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                dfs_bridge(neighbor, depth+1)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] == neighbor and dist[node] < dist[neighbor]:
                    bridges.add((node, neighbor))
            elif neighbor != parent.get(node):
                low[node] = min(low[node], neighbor)

    for node in graph:
        if node not in visited:
            dfs_bridge(node, 0)

    return bridges

def triadicClosure(graph, student1, student2):
    for friend1 in graph[student1]:
        if friend1 in graph[student2]:
            return True
    return False


def plotGraph(graph):

    # create an empty graph
    G = nx.Graph()

    # add edges to the graph
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    # plot the graph
    nx.draw(G, with_labels=True)
    plt.show()



class SchoolNetwork:
    def __init__(self, dataframe):
        self.studentIDs = pd.concat([dataframe["Src"],dataframe["Dst"]]).drop_duplicates().sort_values()
        networkDic = {}
        s = []
        for id in self.studentIDs:
            student = Student(id, dataframe)
            s.append(student)
            networkDic[id] = student.getContact()
        self.network = networkDic
        self.students = s
    
    def getNetwork(self):
        return self.network

    def getStudentIDs(self):
        return self.studentIDs
    
    def getStudents(self):
        return self.students


class Student:

    def __init__(self, id, df):
        self.id = id
        self.dataframe = df[(df["Src"] == id) | (df["Dst"] == id)]

    def __str__(self):
        return "Student ID: " + str(self.id)
    
    def nbrOfMessagesSentTo(self, otherStudentId):
        """
            input: otherStudentId = the id of a other student
            output: the number of message that the student sent to the otherStudentId
        """
        return self.messageSent().loc[self.messageSent() == otherStudentId].size

    def nbrOfMessagesReceivedFrom(self, otherStudentId):
        """
            input: otherStudentId = the id of a other student
            output: the number of message that the student received from the otherStudentId
        """
        return self.messageReceived().loc[self.messageReceived() == otherStudentId].size

    def getId(self):
        return self.id
    
    def messageReceived(self):
        return self.dataframe.loc[self.dataframe["Src"] == self.id, "Dst"]
    
    def messageSent(self):
        return self.dataframe.loc[self.dataframe["Dst"] == self.id, "Src"]
    
    def sentMessageContact(self):
        return self.messageSent().drop_duplicates()
    
    def receivedMessageContact(self):
        return self.messageReceived().drop_duplicates()
    
    def getContact(self):
        return list(pd.concat([self.sentMessageContact(), self.receivedMessageContact()]).drop_duplicates())
    


#graph = network(pd.read_csv("CollegeMsg.csv"))
graph = {
    1: [2, 3],
    2: [1, 3, 8],
    3: [1, 2, 4],
    4: [3, 5],
    5: [4, 6],
    6: [4, 5, 7],
    7 : [6, 8],
    8: [7, 2]
}

# print("nbr of components: ", num_components(graph))
# print("bridges: ", bridges(graph))
# print("local bridges", local_bridge(graph))
print(triadicClosure(graph, 8, 7))
plotGraph(graph)

# school = SchoolNetwork(pd.read_csv("CollegeMsg.csv"))
# print(school.getNetwork())