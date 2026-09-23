
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "online_course_completion.csv")

X = df.drop(columns=["student_id", "course_completed"])
y = df["course_completed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Online Course Completion Prediction")
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred))

joblib.dump(model, BASE / "online_course_completion_model.pkl")

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Online Course Completion - Confusion Matrix")
plt.tight_layout()
plt.savefig(BASE / "confusion_matrix.png")
print("Model and confusion matrix saved.")
