# Decision Tree Regression

# Import the required libraries and the collected dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plotter
collectedDataset=pd.read_csv('JobRole_Salaries.csv')
X=collectedDataset.iloc[:,1:2].values
Y=collectedDataset.iloc[:,2].values

# Split the Dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Fit the Decision Tree Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
decisionTreeRegressor=DecisionTreeRegressor(random_state=0)
decisionTreeRegressor.fit(X,Y)

#Predicting a new value with the trained model
Y_predict=decisionTreeRegressor.predict(6.5)
print(Y_predict)

#Visualize the Decision Tree Regression Results
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plotter.scatter(X,Y,color='green')
plotter.plot(X_grid,decisionTreeRegressor.predict(X_grid),color='red')
plotter.title('Salary Prediction (Decision Tree Regression)')
plotter.xlabel('Job Position level')
plotter.ylabel('Salary')
plotter.show()


