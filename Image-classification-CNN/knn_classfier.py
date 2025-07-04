from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd
# Load the digits dataset
digits=load_digits()
print(digits)
print(digits.data)  
#printing all attributes of the dataset 
print(dir((digits)))
dig = pd.DataFrame(digits.data,digits.target)
dig["target_number"]=digits.target
print(dig)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
score=knn.score(X_test,y_test)

print("score of the model is ",score)
#plotting confussion metrix
confM= confusion_matri
#plotting confussion matrix with heatmap

plt.xlabel("ypredicte")
plt.ylabel("y_test/actual")
plt.title("confusion matrix")
plt.figure(figsize=(7,5))
sns.heatmap(confM, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
print(plt.show())
print("classification report")
print(classification_report(y_test,y_pred))
#plotting some images
def plot_images(x,y,index):
    plt.figure(figsize=(20,3))
    plt.imshow(x[index].reshape(8,8), cmap='gray')
    plt.ylabel(y[index])
    plt.show()  
plot_images(X_test,y_test,4)
