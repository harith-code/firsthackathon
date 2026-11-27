
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("attendance.csv")

# Select input features
features = [
    "total_classes",
    "classes_attended",
    "classes_absent",
    "previous_attendance"
]

X = df[features]

# Target column
y = df["shortage"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

print("Model Evaluation:")
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, "model.pkl")

print("Model saved successfully!")
