import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# load dataset
Dataset = pd.read_csv("NaiveB/spam.csv", encoding='latin-1')
print(Dataset)
#we want all the spam messeges to be 1 and the rest to be 0
Dataset['spam'] = Dataset['Category'].apply(lambda x:1 if x == 'spam' else 0)
print(Dataset)
train_x, test_x, train_y, test_y = train_test_split(Dataset['Message'], Dataset['spam'], test_size=0.2, random_state=42)

# CREATING A PIPELINE TO TEST THE MODEL
PipeObject = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
PipeObject.fit(train_x, train_y)
y_pred = PipeObject.predict(test_x)
print(y_pred)

# Evaluate the model's accuracy
accuracy = accuracy_score(test_y, y_pred)
print(f"Accuracy (Pipeline): {accuracy:.2f}")

# Test with a custom message
email = [" i love you darlie."]
PipeObject.predict(email)
print(PipeObject.predict(email))
if PipeObject.predict(email) == 1:
    print("Spam")
else:
    print("Not Spam")