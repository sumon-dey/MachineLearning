# K- means Clustering

# Import the required libraries and the collected dataset
import pandas as pd
import matplotlib.pyplot as plotter
collectedDataset = pd.read_csv('OnlineBusinessCustomers.csv')
X = collectedDataset.iloc[:, [3, 4]].values

# Use the Elbow Method to find out the optimal number of clusters
from sklearn.cluster import KMeans
wcss = [] # Within Cluster Sum of Squares
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plotter.plot(range(1,11),wcss)
plotter.title('The Elbow Method')
plotter.xlabel('Number of Clusters')
plotter.ylabel('WCSS')
plotter.show()

# Apply the k-means clustering model to the collected dataset
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
Y_kmeans=kmeans.fit_predict(X)

# Visualize the Clusters (for 2-D Clustering)
plotter.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1],s=100,c='red',label='Cluster 1')  
plotter.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1],s=100,c='blue',label='Cluster 2')  
plotter.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1],s=100,c='green',label='Cluster 3')  
plotter.scatter(X[Y_kmeans==3,0],X[Y_kmeans==3,1],s=100,c='cyan',label='Cluster 4')  
plotter.scatter(X[Y_kmeans==4,0],X[Y_kmeans==4,1],s=100,c='magenta',label='Cluster 5')  
plotter.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plotter.title('Clusters of Online Business Customers')
plotter.xlabel('Annual Income (k$)')
plotter.ylabel('Spending Score (1-100)')
plotter.legend()
plotter.show()

# Visualize the Clusters with category names given (for 2-D Clustering)
plotter.scatter(X[Y_kmeans==0,0],X[Y_kmeans==0,1],s=100,c='red',label='Careful')  
plotter.scatter(X[Y_kmeans==1,0],X[Y_kmeans==1,1],s=100,c='blue',label='Standard')  
plotter.scatter(X[Y_kmeans==2,0],X[Y_kmeans==2,1],s=100,c='green',label='Target')  
plotter.scatter(X[Y_kmeans==3,0],X[Y_kmeans==3,1],s=100,c='cyan',label='Careless')  
plotter.scatter(X[Y_kmeans==4,0],X[Y_kmeans==4,1],s=100,c='magenta',label='Sensible')  
plotter.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plotter.title('Clusters of Online Business Customers')
plotter.xlabel('Annual Income (k$)')
plotter.ylabel('Spending Score (1-100)')
plotter.legend()
plotter.show()
