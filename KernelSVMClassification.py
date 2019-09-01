# Kernel Support Vector Machine (SVM)

# Import the required libraries and the collected dataset
import numpy as np
import matplotlib.pyplot as plotter
import pandas as pd
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

# Fit the Kernel SVM to the Training set
from sklearn.svm import SVC
kernelSVMClassifier=SVC(kernel='rbf',random_state=0)
kernelSVMClassifier.fit(X_train,Y_train)

# Predict the Test set results
Y_predict = kernelSVMClassifier.predict(X_test)

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(Y_test, Y_predict)

# Visualize the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plotter.contourf(X1, X2, kernelSVMClassifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plotter.xlim(X1.min(), X1.max())
plotter.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plotter.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plotter.title('Kernel SVM Classification (Training set)')
plotter.xlabel('Age')
plotter.ylabel('Estimated Salary')
plotter.legend()
plotter.show()

# Visualize the Test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plotter.contourf(X1, X2, kernelSVMClassifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plotter.xlim(X1.min(), X1.max())
plotter.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plotter.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plotter.title('Kernel SVM Classification (Test set)')
plotter.xlabel('Age')
plotter.ylabel('Estimated Salary')
plotter.legend()
plotter.show()


