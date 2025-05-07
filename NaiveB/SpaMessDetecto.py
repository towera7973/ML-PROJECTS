import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib as plt

#load dataset 
Dataset = pd.read_csv("NaiveB/spam.csv", encoding='latin-1')
print(Dataset)
Dataset= Dataset[['v1', 'v2']]
Dataset.columns = ['Category', 'message']

Dataset['Category'] = Dataset["Catagory"].map({'ham': 0, 'spam': 1})
print(Dataset)
train_x, train_y,test_x, test_y  = train_test_split(Dataset['Message'],Datase['category'], test_size=0.2, random_state=42)