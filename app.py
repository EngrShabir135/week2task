import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv("iris.data", names=columns)
df.dropna(inplace=True)  # Remove any empty lines

# Streamlit App
st.title("Iris Species Classification using Logistic Regression")
st.write("Upload the Iris dataset or use the default UCI dataset.")

# Display dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Data visualization
st.subheader("Data Visualization")
sns.pairplot(df, hue="species")
plt.savefig("pairplot.png")
st.image("pairplot.png")

# Prepare the dataset
X = df.drop(columns=["species"])
y = df["species"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# User Input for Prediction
st.subheader("Predict Iris Species")
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict Species"):
    input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)
    st.write(f"Predicted Iris Species: {prediction[0]}")