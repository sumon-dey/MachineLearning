# Polynomial Linear Regression

# Import the required libraries and collected dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plotter
collectedDataset = pd.read_csv('JobRole_Salaries.csv')
X = collectedDataset.iloc[:, 1:2].values
Y = collectedDataset.iloc[:, 2].values

# Split the collected dataset into Training set and Test set (with split ratio of 1/5)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fit Simple Linear Regression Model to the dataset
from sklearn.linear_model import LinearRegression
linearRegression1 = LinearRegression()
linearRegression1.fit(X, Y)

# Visualize the Simple Linear Regression Model result
plotter.scatter(X, Y, color = 'green')
plotter.plot(X, linearRegression1.predict(X), color = 'red')
plotter.title('Salary Prediction (Simple Linear Regression)')
plotter.xlabel('Job Position level')
plotter.ylabel('Salary')
plotter.show()

# Fit Polynomial Linear Regression Model to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomialRegression = PolynomialFeatures(degree = 4)
X_polynomial = polynomialRegression.fit_transform(X)
polynomialRegression.fit(X_polynomial, Y)
linearRegression2 = LinearRegression()
linearRegression2.fit(X_polynomial, Y)

# Visualize the Polynomial Linear Regression Model results
plotter.scatter(X, Y, color = 'green')
plotter.plot(X, linearRegression2.predict(polynomialRegression.fit_transform(X)), color = 'red')
plotter.title('Salary Prediction (Polynomial Linear Regression)')
plotter.xlabel('Job Position level')
plotter.ylabel('Salary')
plotter.show()

# Visualize the Polynomial Regression Model results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plotter.scatter(X, Y, color = 'green')
plotter.plot(X_grid, linearRegression2.predict(polynomialRegression.fit_transform(X_grid)), color = 'red')
plotter.title('Salary Prediction (Polynomial Linear Regression)')
plotter.xlabel('Job Position level')
plotter.ylabel('Salary')
plotter.show()

# Predicting and printing a new result with Simple Linear Regression
print(linearRegression1.predict(6.5))

# Predicting and printing a new result with Polynomial Linear Regression
print(linearRegression2.predict(polynomialRegression.fit_transform(6.5)))
