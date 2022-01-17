# COM530500 Network Science Final Project

## 必要的套件 Needed Python Extension Packages (Python 3.9.7)

must import **networkx, numpy , pandas , matplotlib , scipy , random , math** Packages in order to run the program

For **P1.py** you should import :
```python
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
import scipy.io
```

For **P2.py** you should import :
```python
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import scipy.io
import random 
import math
```

## 程式說明 Program Description

本次作業的程式檔案：**P1.py** (For Problem 1) 、**P2.py**（For Problem 2）安裝了必要的套件（如上）直接運行即可，分別完成了Problem 1以及Problem 2 所要求的功能，具體註解已經標註在程式上。**(PS: P1.py 運行時有匯出多張圖像，保存後關閉窗口即可繼續運行。由於使用CPU運算和迴圈多的問題P2.py的運行時間可能會很長。)**
##
**P1.py** (For Problem 1)
  
**Problem 1A**
```python
# Load files get adjacency matrix A

file = scipy.io.loadmat('facebook-ego.mat') 

A = file['A']    

# Read graph

G = nx.from_numpy_matrix(np.matrix(A))  

# Get Number of nodes , Number of edges

n = G.number_of_nodes()
m = G.number_of_edges()

···
# Calculate and print statistical information of this dataset

print("Number of Nodes = ", n)
print("Number of Edges = ", m)
print("Average/Mean Degree = ", mean(a))
print("Max Degree = ", max(a))
print("Diameter = ", nx.diameter(G))
print("Average Clustering Coefficient = ", nx.average_clustering(G))
```
**Problem 1B**
```python
# Visualize the dataset

nx.draw(G , node_size = 10 , node_color = '#00b4d9')   
plt.show()

```
**Problem 1C**
```python

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

···

# Plot the degree distribution with log-log scale

[x,y] = plot_degree_histogram(G)

plt.title('\n Distribution Of Node Linkages (log-log scale) ')
plt.xlabel('Degree \n(log scale)')
plt.ylabel('Number of Nodes \n(log scale)')
plt.xscale("log")
plt.yscale("log")
plt.plot(x, y, 'o')
plt.show()

    
```
**Problem 1D**
```python
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

···

#Print the data
print("Katz Centrality Of Top 10 nodes = ", top_k)
print("Degree Centrality Of Top 10 nodes = ", top_d)
print("Closeness Centrality Of Top 10 nodes = ", top_c)
print("Eigenvector Centrality Of Top 10 nodes = ", top_e)
print("Betweenness Centrality Of Top 10 nodes = ", top_b)
    
```
##
**P2.py** (For Problem 2)

**edge removal function** for removing edges
```python
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

```
**Load file and setting index**

```python

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

```
**Loops for Calculate revalence rate with removing 0%, 10%, 20%, 30%, 40%, 50% of the node**

```python

for c in range(0,5): # loop for caculate every kind of centrity
    print('times =', c) # Runtime counter
    
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
    

```
**Plot the graph**

```python

plt.plot(x_values, r1_array[c], color=colors[c], label = name[c])

ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.legend()
plt.show()

```

(各個程序的運行結果和畫出的圖已保存在result資料夾中)

以上為程式說明，具體的結果報告在 **Network_Science_Final_Project_2021.pdf**

## 參考 Reference
https://stackoverflow.com/questions/65028854/plot-degree-distribution-in-log-log-scale，\
*Methon reference from : M. E. J. Newman, Networks: An
Introduction, Oxford, 2010'*
