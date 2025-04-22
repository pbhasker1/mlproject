import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
def train_model():
    data = pd.read_csv("data.csv")  # Replace with your dataset path
    X = data.drop("target", axis=1)
    y = data["target"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save the trained model
    joblib.dump(model, "pretrained_model.pkl")
    print("Model saved to 'pretrained_model.pkl'")

# Call the function to train the model
if __name__ == "__main__":
    train_model()
