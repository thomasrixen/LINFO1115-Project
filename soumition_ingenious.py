import pandas as pd
from template_utils import *
import matplotlib.pyplot as plt

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
    return sum/len(lcc)


#############################################################################################
#                             first question function                                       #
#############################################################################################

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
def bfs_nodes_path(graph, start, end):
    """
    Finds the shortest path between two nodes in a graph using BFS.
    """
    # Initialize the queue with the start node and the visited set with the start node
    queue = [(start, [start])]
    visited = {start}

    # Loop until the queue is empty
    while queue:
        # Dequeue the next node and its path from the queue
        current_node, path = queue.pop(0)

        # Check if we've reached the end node
        if current_node == end:
            return path

        # Enqueue the neighbors of the current node that haven't been visited yet
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    # If we reach this point, there's no path between the start and end nodes
    return None


def count_local_bridges(graph):
    """
    Computes the number of local bridges in a graph represented as a dictionary.
    """
    count = 0

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            graph[node].remove(neighbor)
            graph[neighbor].remove(node)
            short_path = bfs_nodes_path(graph, node, neighbor)
            if short_path == None:
                graph[node].append(neighbor)
                graph[neighbor].append(node)
                count += 1
            else:
                distance_after = len(short_path)-1
                graph[node].append(neighbor)
                graph[neighbor].append(node)
                if distance_after > 2:
                    count += 1

    return count//2



#############################################################################################
#                             second question function                                      #
#############################################################################################
def count_triadic_closures(last_messages_df):
    # Compute the median timestamp
    median_time = last_messages_df['Time'].median()

    # Filter the DataFrame to include only messages sent after the median timestamp
    filtered_df = last_messages_df[last_messages_df['Time'] >= median_time]

    # Create a dictionary to store the nodes and their neighbors
    nodes = {}
    for index, row in filtered_df.iterrows():
        sender, receiver = row['Src'], row['Dst']
        if sender not in nodes:
            nodes[sender] = set()
        if receiver not in nodes:
            nodes[receiver] = set()
        nodes[sender].add(receiver)
        nodes[receiver].add(sender)

    # Compute the number of triadic closures that have appeared between the median timestamp and the end
    count = 0
    for node in nodes:
        neighbors = nodes[node]
        for neighbor in neighbors:
            for neighbor2 in neighbors:
                if neighbor != neighbor2 and neighbor2 in nodes[neighbor]:
                    count += 1
    return count

# Plot the accumulated number of triadic closures over time
def plot_triadic_closures(last_messages_df):
    # Sort the DataFrame by timestamp in ascending order
    sorted_df = last_messages_df.sort_values('Time', ascending=True)

    # Create an empty list to store the accumulated number of triadic closures
    counts = []

    # Iterate over the messages in the DataFrame and compute the number of triadic closures after each message
    for i in range(len(sorted_df)):
        # Compute the number of triadic closures after the i-th message
        count = count_triadic_closures(sorted_df.iloc[:i+1])
        print(count)
        # Add the count to the list of accumulated counts
        counts.append(count)

    # Plot the accumulated number of triadic closures over time
    plt.plot(sorted_df['Time'], counts)
    plt.xlabel('Time')
    plt.ylabel('Number of triadic closures')
    plt.show()
    
    #############################################################################################
#                             thirde question function                                       #
#############################################################################################

def bfs_small_paths(graph, start_node, small_paths_list):
    # Initialize the queue with the starting node and a path of length 0
    queue = [(start_node, 0)]
    visited = set()

    # BFS to find the small paths for the starting node
    while queue:
        curr_node, curr_dist = queue.pop(0)
        if curr_node not in visited:
            visited.add(curr_node)
            if curr_dist > 0 and curr_dist < len(small_paths_list):
                small_paths_list[curr_dist] += 1
            for neighbor in graph[curr_node]:
                queue.append((neighbor, curr_dist+1))

    return small_paths_list

#############################################################################################
#                             forth question function                                      #
#############################################################################################
def pagerank(graph, d=0.85, tol=1e-10):
    # Initialize PageRank values
    N = len(graph)
    pagerank_dict = {node: 1 / N for node in graph}
    pagerank_dict_old = pagerank_dict.copy()

    # Perform iterative calculation until convergence
    while True:
        # Calculate PageRank for each node
        for node in pagerank_dict:
            B = [n for n in graph if node in graph[n]]
            if len(B) > 0:
                pagerank_dict[node] = (1 - d) / len(graph) + d * sum(
                    pagerank_dict[n] / len(graph[n]) for n in B
                )

        # Check for convergence
        max_diff = max(abs(pagerank_dict[node] - pagerank_dict_old[node]) for node in pagerank_dict)
        if max_diff < tol:
            break
        pagerank_dict_old = pagerank_dict.copy()
    counter = 0
    counter2 = 0
    for i in pagerank_dict.values():
        counter+= i
    for i in range(len(pagerank_dict)):
        pagerank_dict[i] /= counter

    return pagerank_dict

#############################################################################################
#                             fifth question function                                      #
#############################################################################################
def local_clustering_coefficient(graph):
    clustering_coefficient = {}
    for node in graph:
        neighbors = graph[node]
        degree = len(neighbors)
        if degree < 2:
            clustering_coefficient[node] = 0.0
            continue
        triangles = 0
        for neighbor1 in neighbors:
            for neighbor2 in neighbors:
                if neighbor1 != neighbor2 and neighbor1 in graph[neighbor2]:
                    triangles += 1
        clustering_coefficient[node] = (2 * triangles) / (degree * (degree - 1))
    return clustering_coefficient

#############################################################################################
#                              Helper classes                                               #
#############################################################################################

class SchoolNetwork:
    def __init__(self, dataframe):
        self.studentIDs = pd.concat([dataframe["Src"],dataframe["Dst"]]).drop_duplicates().sort_values()
        networkDic = {}
        directedNetworkDic = {}
        s = []
        for id in self.studentIDs:
            student = Student(id, dataframe)
            s.append(student)
            networkDic[id] = student.getContact()
            directedNetworkDic[id] = list(student.getSentMessageContact())
        self.network = networkDic
        self.directedNetwork = directedNetworkDic
        self.students = s
        last_messages = {}
        for index, row in dataframe.iterrows():
            sender, receiver, timestamp = row['Src'], row['Dst'], row['Time']
            if (sender, receiver) not in last_messages and (receiver, sender) not in last_messages:
                last_messages[(sender, receiver)] = row
        self.last_messages_df = pd.DataFrame(last_messages.values(), columns=['Src', 'Dst', 'Time'])
    
    def getNetwork(self):
        return self.network
    
    def getDirectedNetwork(self):
        return self.directedNetwork

    def getStudentIDs(self):
        return self.studentIDs
    
    def getStudents(self):
        return self.students
    
    def getLastMessage(self):
        return self.last_messages_df


class Student:

    def __init__(self, id, df):
        self.id = id
        self.dataframe = df[(df["Src"] == id) | (df["Dst"] == id)]
        self.messageReceived = self.dataframe.loc[self.dataframe["Src"] == self.id, "Dst"]
        self.messageSent = self.dataframe.loc[self.dataframe["Dst"] == self.id, "Src"]
        self.sentMessageContact = self.messageSent.drop_duplicates()
        self.receivedMessageContact = self.messageReceived.drop_duplicates()
        self.contact = list(pd.concat([self.sentMessageContact, self.receivedMessageContact]).drop_duplicates())

    def __str__(self):
        return "Student ID: " + str(self.id)
    
    def nbrOfMessagesSentTo(self, otherStudentId):
        """
            input: otherStudentId = the id of a other student
            output: the number of message that the student sent to the otherStudentId
        """
        return self.getMessageSent().loc[self.getMessageSent() == otherStudentId].size

    def nbrOfMessagesReceivedFrom(self, otherStudentId):
        """
            input: otherStudentId = the id of a other student
            output: the number of message that the student received from the otherStudentId
        """
        return self.getMessageReceived().loc[self.getMessageReceived() == otherStudentId].size

    def getId(self):
        return self.id
    
    def getMessageReceived(self):
        return self.messageReceived
    
    def getMessageSent(self):
        return self.messageSent
    
    def getSentMessageContact(self):
        return self.sentMessageContact
    
    def getReceivedMessageContact(self):
        return self.receivedMessageContact
    
    def getContact(self):
        return self.contact
    
df = pd.read_csv('CollegeMsg.csv')
print(Q1(df))
print(Q2(df))
print(Q3(df))
print(Q4(df))
print(Q5(df))