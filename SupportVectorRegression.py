#Support Vector Regression

# Import the required libraries and the collected dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plotter
collectedDataset=pd.read_csv('JobRole_Salaries.csv')
X=collectedDataset.iloc[:,1:2].values
Y=collectedDataset.iloc[:,2:3].values

# Split the Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Perform Featue Scaling of the collected dataset
from sklearn.preprocessing import StandardScaler
standardScaler_X=StandardScaler()
standardScaler_Y=StandardScaler()
X=standardScaler_X.fit_transform(X)
Y=standardScaler_Y.fit_transform(Y)

# Fit the Support Vector Regression Model to the dataset
from sklearn.svm import SVR
supportVectorRegressor=SVR(kernel='rbf')
supportVectorRegressor.fit(X,Y)

# Predict a new Result with the built SVR model
Y_predicted=standardScaler_Y.inverse_transform(supportVectorRegressor.predict(standardScaler_X.transform(np.array([[6.5]]))))
print(Y_predicted)

#Visualize the SVR Results
plotter.scatter(X,Y,color='green')
plotter.plot(X,supportVectorRegressor.predict(X),color='red')
plotter.title('Salary Prediction (Support Vector Regression)')
plotter.xlabel('Job Position level')
plotter.ylabel('Salary')
plotter.show()




