import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LinearRegression
model=LinearRegression()

HP_pdfile=pd.read_csv('HP_detector_with_OHE/homeprices.csv')
print(HP_pdfile)

#using labelencoder library from sklearn to nncode one and zero on town data 
from sklearn.preprocessing import LabelEncoder
modelLe=LabelEncoder()     #creating object for data encoding 

HP_pdfile.town = modelLe.fit_transform(HP_pdfile.town)    #automaticall encoding the data to numbers instead of names  on town data 
print("encoded town data")
print(HP_pdfile.town)

#N ow we want to remove price ,,but we will use a method that is going to extract all data except price unlike with dummies method which uses drop method

X = HP_pdfile[['town','area']].values
print(X)

#putting the output in y 
y=HP_pdfile.price.values
print(y)
#converting town values to dummy columns values with onehotencoder 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
modelCT=ColumnTransformer([('town',OneHotEncoder(),[0])],remainder='passthrough')
encoded_X = modelCT.fit_transform(X)
print(encoded_X)
#we can drop one column of town 
encoded_X=encoded_X[:,1:]
print("removed one town data to be virtually seen")
print(encoded_X)

model.fit(encoded_X,y)
predictedPrice=model.predict([[1,0,200],[0,1,5999]])
print("The Predicted price for the two arrays of data is ")
print(predictedPrice)



