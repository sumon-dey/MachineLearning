# Random Forest Regression

# Import the required libraries and the collected dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plotter
collectedDataset=pd.read_csv('JobRole_Salaries.csv')
X=collectedDataset.iloc[:,1:2].values
Y=collectedDataset.iloc[:,2].values

# Split the collected dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Fit the Random Forest Regression Model to the dataset with n_estimators=10,
# where n_estimators is the number of trees in the forest
from sklearn.ensemble import RandomForestRegressor
randomForestRegressor=RandomForestRegressor(n_estimators=10,random_state=0)
randomForestRegressor.fit(X,Y)

# Fit the Random Forest Regression Model to the dataset
# from sklearn.ensemble import RandomForestRegressor
# randomForestRegressor=RandomForestRegressor(n_estimators=100,random_state=0)
# randomForestRegressor.fit(X,Y)

# Fit the Random Forest Regression Model to the dataset
# from sklearn.ensemble import RandomForestRegressor
# randomForestRegressor=RandomForestRegressor(n_estimators=300,random_state=0)
# randomForestRegressor.fit(X,Y)


# Predict and print the output for a new data based on the formed Random Forest Regression Model
Y_predict=randomForestRegressor.predict(6.5)
print(Y_predict)

# Visualize the Random Forest Regression Model Results
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plotter.scatter(X,Y,color='green')
plotter.plot(X_grid,randomForestRegressor.predict(X_grid),color='red')
plotter.title('Salary Prediction (Random Forest Regression)')
plotter.xlabel('Job Position level')
plotter.ylabel('Salary')
plotter.show()


