# Text Classification using Naive Bayes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Step 2: Load the dataset
df = pd.read_csv("synthetic_text_data.csv")
x = df['text']
y = df['label']

# Step 3: Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorsizer = CountVectorizer()
x_train_vectorsized = vectorsizer.fit_transform(x_train)
x_test_vectorsized = vectorsizer.transform(x_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(x_train_vectorsized, y_train)

# Predictions
y_pred = model.predict(x_test_vectorsized)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Classification Report:", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y_test), 
            yticklabels=np.unique(y_test))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Prediction on unseen data
user_input = "AI is my favourite technology"
user_input_vectorsized = vectorsizer.transform([user_input])
prediction = model.predict(user_input_vectorsized)
print(f'Prediction for the new input: {prediction[0]}')
