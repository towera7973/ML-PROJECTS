import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
Iris_data= load_iris()
#print(Iris_data.data)
#print(Iris_data.target_names)
#print(Iris_data.feature_names)
#reading data with its feature names the convinient way with pandas
Iris_pd=pd.DataFrame(Iris_data.data, columns=Iris_data.feature_names)
print(Iris_pd.head())
#show the target aswel (the digits of the flowers ,e.g 0.1.2) by appending columns
Iris_pd["Target"]=Iris_data.target
print(Iris_pd)

X = Iris_pd.drop('Target', axis=1)  # Features (all columns except 'Target')
y = Iris_pd['Target']              # Target variable ('Target' column)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=42) # Added random_state for reproducibility

print(len(X_train)) 
print(y_train.head())
#No using svc to train data
model=SVC()
model.fit(X_train,y_train)
accurency=model.score(X_test,y_test)
print(accurency)
y_predicted=model.predict(X_test)
print(y_predicted)
flowernumber=model.predict([[9.4,3.9,7.7,4]])
print(flowernumber)
flowername=Iris_data.target_names[flowernumber]
print(flowername)