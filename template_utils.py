#if needed, write here your additional fuctions/classes with their signature and use them in the exercices:
# a specific place is available to copy them at the end of the Inginious task.

#First, import the libraries needed for your helper functions
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


def network(dataframe):
    networkDic = {}
    studentIDs = pd.concat([dataframe["Src"],dataframe["Dst"]]).drop_duplicates().sort_values()
    for id in studentIDs:
        networkDic[id] = studentContact(id, dataframe)
    return networkDic



def studentContact(id, dataframe):
    messageReceived = dataframe.loc[dataframe["Src"] == id, "Dst"]
    messageSent = dataframe.loc[dataframe["Dst"] == id, "Src"]
    sentMessageContact = messageSent.drop_duplicates()
    receivedMessageContact = messageReceived.drop_duplicates()
    
    return list(pd.concat([sentMessageContact, receivedMessageContact]).drop_duplicates())




# class School:
#     def __init__(self, dataframe):
#         self.studentIDs = pd.concat([dataframe["Src"],dataframe["Dst"]]).drop_duplicates().sort_values()
#         self.students = []
#         for id in self.studentIDs:
#             self.students.append(Student(id, dataframe))
#         self.size = len(self.students)

#     def __sizeof__(self):
#         return self.size
    
#     def getStudents(self):
#         return self.students

#     def getStudentIDs(self):
#         return self.studentIDs
        


# class Student:

#     def __init__(self, id, dataframe):
#         self.id = id
#         self.messageReceived = dataframe.loc[dataframe["Src"] == id, "Dst"]
#         self.messageSent = dataframe.loc[dataframe["Dst"] == id, "Src"]
#         self.sentMessageContact = self.messageSent.drop_duplicates()
#         self.receivedMessageContact = self.messageReceived.drop_duplicates()
#         self.contact = pd.concat([self.sentMessageContact, self.receivedMessageContact]).drop_duplicates()

#     def __str__(self):
#         return str(self.id)
    
#     def nbrOfMessagesSentTo(self, otherStudentId):
#         """
#             input: otherStudentId = the id of a other student
#             output: the number of message that the student sent to the otherStudentId
#         """
#         return self.getMessageSent().loc[self.getMessageSent() == otherStudentId].size

#     def nbrOfMessagesReceivedFrom(self, otherStudentId):
#         """
#             input: otherStudentId = the id of a other student
#             output: the number of message that the student received from the otherStudentId
#         """
#         return self.getMessageReceived().loc[self.getMessageReceived() == otherStudentId].size

#     def getId(self):
#         return self.id
    
#     def getMessageReceived(self):
#         return self.messageReceived
    
#     def getMessageSent(self):
#         return self.messageSent
    
#     def getSentMessageContact(self):
#         return self.sentTo
    
#     def getReceivedMessageContact(self):
#         return self.receivedFrom
    
#     def getContact(self):
#         return self.contact
    
#

#graph = network(pd.read_csv("CollegeMsg.csv"))
graph = {
    1: [2, 3],
    2: [1, 3],
    3: [1, 2, 4],
    4: [3, 5],
    5: [4, 6],
    6: [4, 5, 7],
    7 : [6, 8],
    8: [7]
}
print("nbr of components: ", num_components(graph))
print("bridges: ", bridges(graph))