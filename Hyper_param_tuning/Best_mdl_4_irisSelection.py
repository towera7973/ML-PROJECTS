from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
# Importing the necessary libraries

IrisData= load_iris()
print("Description of the Iris Dataset:\n", IrisData.DESCR)
print(IrisData)
iris=pd.DataFrame(data=IrisData.data, columns=IrisData.feature_names)
print(iris.head())
#appending the target column to the iris dataframe
iris['Targets']=IrisData.target
print(iris.head())
#assigning target name to the target column
iris['Targets']=iris['Targets'].apply(lambda x: IrisData.target_names[x])
print(iris.head())

x_train, x_test, y_train, x_test = train_test_split(IrisData.data ,IrisData.target , test_size=0.2, random_state=42)
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)

#Creating a dictionary  with all models and their parameters
models_params_dict = {
    'SVC': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': [0.1, 1, 10]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20]
        }
    }
}

score_dict = []
for model_name ,mp in models_params_dict.items():
    classifier= GridSearchCV(mp["model"], mp["params"], cv=5, return_train_score=False)
    classifier.fit(x_train, y_train)

    score_dict.append(
        {
            'model': model_name,
            'best_score': classifier.best_score_,
            'best_params': classifier.best_params_

        }
    )
score= pd.DataFrame(score_dict)
print(score)