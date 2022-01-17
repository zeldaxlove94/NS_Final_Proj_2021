import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import scipy.io
import random 
import math

np.set_printoptions(threshold = 3000)

def edge_removal(A,phi):
    '''Remove edges by the trick developed by Mollison and Grassberger
    Args:
        A: numpy.ndarray
            Adjacency matrix of the dataset
        phi: float
            prob. of the edge is present

    Returns:
        A_new: numpy.ndarray
            Adjacency matrix after edge removal
    '''
    P = np.triu(np.random.random_sample(A.shape),1)
    Prob = P+P.T 
    A_new = ((A * Prob)> 1-phi).astype(int)
    return A_new


# Load files

file = scipy.io.loadmat('facebook-ego.mat')

# Set adjacency matrix A

A = file['A']  

# Read Graph

G1 = nx.from_numpy_matrix(np.matrix(A))

r1_array = [[],[],[],[],[]] # Creat r1 array for store r1 value
per = [0, 0.1, 0.2, 0.3, 0.4, 0.5] # Set removing percentages
cent_name = ["Degree Centrality", "Katz Centrality", "Eigenvector Centrality", "Betweenness Centrality", "Closeness Centrality"] # Names of Centrality

# Set graph index

x_major_locator = MultipleLocator(10)
x_values = [0, 10, 20, 30, 40, 50] 
colors = ['r', 'g', 'b', 'y', 'm'] 

plt.figure(1)
plt.clf()
plt.xlabel('percentage(%)')
plt.ylabel('r1')

for c in range(0,5): # loop for caculate every kind of centrity
    #print('times =', c) # Runtime counter
    
    for r in range(0,6): # 6 removing rate
        
        sum = 0
        avg = 0

        for t in range(1,3): # run 100 times 
            
            # Set removing rate

            A_reml = edge_removal(A, 0.1)
            
            # Read graph

            G2 = nx.from_numpy_matrix(np.matrix(A_reml)) 
            
            # Get centrality

            if c == 0:cent = nx.degree_centrality(G2)
            if c == 1:cent = nx.katz_centrality_numpy(G2)
            if c == 2:cent = nx.eigenvector_centrality(G2)
            if c == 3:cent = nx.betweenness_centrality(G2)
            else:cent = nx.closeness_centrality(G2)
            
            # Sort centrality

            centrality = (sorted(cent.items(), key = lambda x:x[1], reverse = True)) 
            i = math.floor(per[r]*len(centrality))
          
            for i in range(0,i):
                s = np.array(centrality)[i][0]
               
                for num in range(0, n):
                    A_reml[int(s)][num] = 0
                    A_reml[num][int(s)] = 0
        
            # Calculate Y[]

            l = n = G2.number_of_nodes() # length = n
            D = nx.diameter(G1)
            na = np.array([i for i in range(int(l))])
            x = np.zeros(int(l))

            for i in random.sample(na.tolist(), 5): # Random chosen 5 nodes
                x[i] = 1 # set 1 as infected
            
            y = ((A_reml + np.eye(l))**D) # Get Y[]
            f = np.matmul(y,x)
            
            # Calculate number of infected nodes
            
            count = 0
            for j in range(0, n):
                if f[j] == 1:
                 count += 1
            sum += count
        

        # Get average of r1 values
        
        avg = sum/(t-1)

        r1_value = avg*100/n

        r1_array[c].append(r1_value)

        # Print the result and plot the graph

        print(f"{cent_name[c]} prevalence rate with removing {100*per[r]}% of the node: {(r1_value)} (%)")
    
    plt.plot(x_values, r1_array[c], color=colors[c], label = cent_name[c])

ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.legend()
plt.show()
