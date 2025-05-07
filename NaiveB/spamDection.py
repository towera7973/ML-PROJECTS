import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset (replace 'spam.csv' with your dataset path)
# Assuming the dataset has 'label' and 'message' columns
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['Category', 'message']

# Encode labels: 'spam' -> 1, 'ham' -> 0
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['category'], test_size=0.2, random_state=42
)

# Convert text data to feature vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Test with a custom message
def predict_spam(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example usage
print(predict_spam("Congratulations! You've won a free lottery ticket."))