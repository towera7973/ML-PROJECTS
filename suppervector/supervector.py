import pandas as pd
import matplotlib as plt
from sklearn.datasets import load_iris
Iris_data= load_iris()
print(Iris_data.data)
print(Iris_data.target_names)
print(Iris_data.feature_names)
#reading data with its feature names the convinient way with pandas
Iris_data=pd.DataFrame(Iris_data.data, columns=Iris_data.feature_names)
print(Iris_data.head())
#show the target aswel (the digits of the flowers ,e.g 0.1.2




