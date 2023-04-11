import pandas as pd
import numpy as np
from templates_utils import *
import sys 
sys.setrecursionlimit(6000)

def Q1(dataframe):
    #Your code here
    return [2, 80, 112] #[nb connected components, nb bridges, nb local bridges]

def Q2(dataframe):
    #Your code here
    return 192384 #total number of triadic closures created after the median timestamp

def Q3(dataframe):
    #Your code here
    return [32, 19, 32, 29, 12] #at index 0 the number of shortest paths of lenght 0, at index 1 the number of shortest paths of length 1, ...

def Q4(dataframe):
    #Your code here
    return [10, 0.2413] # the id of the node with the highest pagerank score, the associated pagerank value.
    #Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-10)

def Q5(dataframe):
    #Your code here
    return 0.5555 #the average local clustering coefficient of the graph

#you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

df = pd.read_csv('CollegeMsg.csv')
print(Q1(df))
print(Q2(df))
print(Q3(df))
print(Q4(df))
print(Q5(df))
