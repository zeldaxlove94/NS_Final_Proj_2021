import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
import scipy.io

# plot_degree_histogram function
# method reference from : https://stackoverflow.com/questions/65028854/plot-degree-distribution-in-log-log-scale

def plot_degree_histogram(g, normalized=True):
    print("Creating histogram...")
    aux_y = nx.degree_histogram(g)
    
    aux_x = np.arange(0,len(aux_y)).tolist()
    
    n_nodes = g.number_of_nodes()
    
    if normalized:
        for i in range(len(aux_y)):
            aux_y[i] = aux_y[i]/n_nodes
    
    return aux_x, aux_y


# Load files get adjacency matrix A

file = scipy.io.loadmat('facebook-ego.mat') 

A = file['A']  

# Read graph

G = nx.from_numpy_matrix(np.matrix(A))  

# Get Number of nodes , Number of edges

n = G.number_of_nodes()
m = G.number_of_edges()


# Calculate katz,degree,closeness,eigenvector,betweenness centrality and get top ten of each if centrality

katz = nx.katz_centrality_numpy(G)
kc = (sorted(katz.items(),key=lambda x:x[1],reverse = True))  # Sort items (High to low) and get to ten of it                    
top_k = kc[0:10]                                                
d = nx.degree_centrality(G)                                  
dc = (sorted(d.items(),key=lambda x:x[1],reverse = True))  
top_d = dc[0:10]      
c = nx.closeness_centrality(G)                               
cc = (sorted(c.items(),key=lambda x:x[1],reverse = True))
top_c = cc[0:10]                                            
e = nx.eigenvector_centrality(G)
ec = (sorted(e.items(),key=lambda x:x[1],reverse = True))
top_e = ec[0:10]
b = nx.betweenness_centrality(G)
bc = (sorted(b.items(),key=lambda x:x[1],reverse = True))
top_b = bc[0:10]

a = dict(G.degree())
a = list(a.values())

# Plot the degree distribution with log-log scale

[x,y] = plot_degree_histogram(G)

plt.title('\n Distribution Of Node Linkages (log-log scale) ')
plt.xlabel('Degree \n(log scale)')
plt.ylabel('Number of Nodes \n(log scale)')
plt.xscale("log")
plt.yscale("log")
plt.plot(x, y, 'o')
plt.show()

# Calculate and print statistical information of this dataset

print("Number of Nodes = ", n)
print("Number of Edges = ", m)
print("Average/Mean Degree = ", mean(a))
print("Max Degree = ", max(a))
print("Diameter = ", nx.diameter(G))
print("Average Clustering Coefficient = ", nx.average_clustering(G))

print("Katz Centrality Of Top 10 nodes = ", top_k)
print("Degree Centrality Of Top 10 nodes = ", top_d)
print("Closeness Centrality Of Top 10 nodes = ", top_c)
print("Eigenvector Centrality Of Top 10 nodes = ", top_e)
print("Betweenness Centrality Of Top 10 nodes = ", top_b)

# Visualize the dataset

nx.draw(G , node_size = 10 , node_color = '#00b4d9')   
plt.show()




