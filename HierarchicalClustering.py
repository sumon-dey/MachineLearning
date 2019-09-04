# Hierarchical Clustering

# Import the required libraries and the collected dataset
import matplotlib.pyplot as plotter
import pandas as pd
collectedDataset = pd.read_csv('OnlineBusinessCustomers.csv')
X = collectedDataset.iloc[:,[3,4]].values

# Use Dendrogram to find out the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plotter.title('Dendrogram')
plotter.xlabel('OnlineBusinessCustomers')
plotter.ylabel('Euclidean Distances')
plotter.show()

# Fit the Hierarchical Clustering Model to the dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
Y_hc=hc.fit_predict(X)

# Visualize the clusters
plotter.scatter(X[Y_hc==0,0],X[Y_hc==0,1],s=100,c='red',label='Careful')  
plotter.scatter(X[Y_hc==1,0],X[Y_hc==1,1],s=100,c='blue',label='Standard')  
plotter.scatter(X[Y_hc==2,0],X[Y_hc==2,1],s=100,c='green',label='Target')  
plotter.scatter(X[Y_hc==3,0],X[Y_hc==3,1],s=100,c='cyan',label='Careless')  
plotter.scatter(X[Y_hc==4,0],X[Y_hc==4,1],s=100,c='magenta',label='Sensible')
plotter.title('Clusters of OnlineBusinessCustomers')
plotter.xlabel('Annual Income (k$)')
plotter.ylabel('Spending Score (1-100)')
plotter.legend()
plotter.show()
 
