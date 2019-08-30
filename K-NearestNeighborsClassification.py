#KNN (K-Nearest Neighbors) Classification

# Import the required libraries and collected dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plotter
collectedDataset = pd.read_csv('Car_Purchase.csv')
X = collectedDataset.iloc[:, [2, 3]].values
Y = collectedDataset.iloc[:, 4].values

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scale
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)

# Fit the KNN classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,Y_train)

# Predict the Test set results
Y_pred = classifier.predict(X_test)

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Visualize the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plotter.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plotter.xlim(X1.min(), X1.max())
plotter.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plotter.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plotter.title('K-NN (Training set)')
plotter.xlabel('Age')
plotter.ylabel('Estimated Salary')
plotter.legend()
plotter.show()

# Visualize the Test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plotter.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plotter.xlim(X1.min(), X1.max())
plotter.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plotter.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plotter.title('K-NN (Test set)')
plotter.xlabel('Age')
plotter.ylabel('Estimated Salary')
plotter.legend()
plotter.show()




