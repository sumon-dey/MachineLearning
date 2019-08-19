# Multiple Linear Regression

# Import the required pandas library and collected companies dataset
import pandas as pd
collectedDataset = pd.read_csv('50-startups.csv')
X = collectedDataset.iloc[:, :-1].values
Y = collectedDataset.iloc[:, 4].values

# Encode the Categorical data to convert to Continuous Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:, 3] = labelEncoder.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()

# Avoid the Dummy Variable Trap
X = X[:, 1:]

# Split the collected dataset into the Training set and Test set (with split ratio 1/3)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


# Fit the Multiple Linear Regression model to the Training set
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(X_train, Y_train)

# Predict the Test set results and compare with Y_test
Y_pred = mlr.predict(X_test)
print (Y_pred)
print (Y_test)

