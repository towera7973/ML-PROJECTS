import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


Dt1=pd.read_csv("House_price_prediction_pjct/model/bengaluru_house_prices.csv")
Data_1=pd.DataFrame(Dt1)
print(Data_1.head())
print(Data_1.columns)
print(Data_1.shape)
print(Data_1.area_type.unique())

#data cleaning,dropping columns not wanted 
Data_1.drop(['area_type', 'availability', 'society', 'balcony'] , axis='columns', inplace=True)
print(Data_1.shape)
print(Data_1.head())
#data cleaning 
print(Data_1.isnull().sum())
#drop all the rows with null values 
Data_2=Data_1.dropna()
print(Data_2.isnull().sum())

#now we have to intergerise the size column to number and BHK ,BEDROON HOUSE KITCHEN, BATHROOM
Data_3=Data_2.copy()
Data_3['Bedrooms']= Data_3['size'].apply( lambda x:  int(x.split(' ')[0]))
print(Data_3.head())
#we can now drop size column
Data_4=Data_3.drop(['size'] , axis='columns')
print(Data_4.head())

