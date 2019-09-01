#Simple Linear Regression

#Import the required libraries and the collected Defect Dataset from csv file
import pandas as pd
import matplotlib.pyplot as plotter
collectedDataset = pd.read_csv('MonthlyDefects_Data.csv')

#Create Matrix of features for independent variable-X(Test Execution Duration(in Months)) 
#and for dependent variable-Y(No. of Defects Found))
X=collectedDataset.iloc[:,:-1].values
Y=collectedDataset.iloc[:,1].values

#Split the collected dataset into Training set and Test set (with split ratio 1/3)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


#Fit the Simple Linear Regression model to the Training set
from sklearn.linear_model import LinearRegression
slr=LinearRegression()
slr.fit(X_train,Y_train)

#Predict the Test Set Results and a sample value of independent variable
#Create a vector of predictions of the Dependent variable
Y_pred=slr.predict(X_test)
sample=slr.predict(4.2)
print(sample)

#Visualize the Training set result
plotter.scatter(X_train,Y_train,color='green')
plotter.plot(X_train,slr.predict(X_train),color='red')
plotter.title('Defects Found vs Months of Execution (Training Set)')
plotter.xlabel('Months of Execution')
plotter.ylabel('Defects Found')
plotter.show()

#Visualizing the Test set result
plotter.scatter(X_test,Y_test,color='green')
plotter.plot(X_train,slr.predict(X_train),color='red')
plotter.title('Defects Found vs Months of Execution (Test Set)')
plotter.xlabel('Months of Execution')
plotter.ylabel('Defects Found')
plotter.show()

