# Logistic Regression

# Import the required libraries and the collected dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plotter
collectedDataset=pd.read_csv('Social_Network_Ads.csv')
X=collectedDataset.iloc[:,[2,3]].values
Y=collectedDataset.iloc[:,4].values

# Split the Dataset into the Training Set and Test Set with test size of 0.2
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Perform Featue Scaling on the data points
from sklearn.preprocessing import StandardScaler
standardScaler_X=StandardScaler()
X_train=standardScaler_X.fit_transform(X_train)
X_test=standardScaler_X.transform(X_test)

# Fit the Logistic Regression Model to the Training Set
from sklearn.linear_model import LogisticRegression
logisticRegressionClassifier=LogisticRegression(random_state=0) 
logisticRegressionClassifier.fit(X_train,Y_train)

# Predict the Test Set Results bqsed on the model
Y_predict=logisticRegressionClassifier.predict(X_test)

# Create the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_predict)

# Visualize the Training Set Results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plotter.contourf(X1, X2, logisticRegressionClassifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plotter.xlim(X1.min(), X1.max())
plotter.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plotter.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plotter.title('Logistic Regression (Training set)')
plotter.xlabel('Age')
plotter.ylabel('Estimated Salary')
plotter.legend()
plotter.show()

# Visualize the Test Set Results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plotter.contourf(X1, X2, logisticRegressionClassifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plotter.xlim(X1.min(), X1.max())
plotter.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plotter.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plotter.title('Logistic Regression (Test set)')
plotter.xlabel('Age')
plotter.ylabel('Estimated Salary')
plotter.legend()
plotter.show()
  
