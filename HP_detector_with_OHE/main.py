import pandas as pd # Imports the pandas library for data manipulation.
HP_pdfile=pd.read_csv("HP_detector_with_OHE/homeprices.csv") # into a pandas DataFrame.

print(HP_pdfile) # Displays the initial DataFrame.
Dummies =pd.get_dummies(HP_pdfile.town).astype(int) # Creates dummy variables for the 'town' column. astype(int) converts the boolean (True/False) values to integers (1/0).
print(Dummies) # Displays the dummy variable DataFrame.
MergedDT=pd.concat([HP_pdfile,Dummies], axis="columns") #Merges the original DataFrame with the dummy variable DataFrame along the columns.
print(MergedDT)# Displays the merged DataFrame.
Drop_Town=MergedDT.drop(['town'], axis='columns')   # Removes the original 'town' column.
print(Drop_Town)  #Displays the DataFrame after dropping the 'town' column.
Drop_Variable=Drop_Town.drop(['west windsor'], axis='columns')   # Drops one of the dummy variable columns ('west windsor') to avoid multicollinearity (the dummy variable trap). This is a good practice.
print(Drop_Variable)         # Displays the DataFrame after dropping 'west windsor'.
x=Drop_Variable.drop(['price'], axis='columns')   # Creates the feature matrix x by dropping the 'price' column. These are the independent variables used for prediction.
print(x)    #Displays the feature matrix x.
y= Drop_Variable.price   #Creates the target variable y (the 'price' column), which is what we want to predict.
print(y)              # Displays the target variable y.
from sklearn.linear_model import LinearRegression     #Imports the LinearRegression class from scikit-learn.
model=LinearRegression()       #Creates an instance of the LinearRegression model.
model.fit(x,y)                 #Trains the linear regression model using the feature matrix x and the target variable y.
model.predict([[3000,0,1]])    #Potential Issue: This line attempts to predict the price of a 3000 sq ft house in "robinsville"