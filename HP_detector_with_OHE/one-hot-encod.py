import pandas as pd
import matplotlib as plt

HP_pdfile=pd.read_csv('HP_detector_with_OHE/homeprices.csv')
print(HP_pdfile)

#using labelencoder library from sklearn to nncode one and zero on town data 
from sklearn.preprocessing import LabelEncoder
modelLe=LabelEncoder()     #creating object for data encoding 
HP_pdfile.town = modelLe.fit_transform(HP_pdfile.town)    #automaticall encoding the data with 0 and 1 on town data 
print(HP_pdfile.town)

#N ow we want to remove price ,,but we will use a method that is going to extract all data except price unlike with dummies method which uses drop method

X = HP_pdfile[['town','area']].values
print(X)

