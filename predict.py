
from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / "online_course_completion_model.pkl")

print("Enter student details:")
age = int(input("Age: "))
study_hours = float(input("Weekly study hours: "))
videos = int(input("Videos watched: "))
assignments = int(input("Assignments submitted: "))
forum_posts = int(input("Forum posts: "))
login_days = int(input("Login days: "))
previous_score = float(input("Previous score: "))

sample = pd.DataFrame([{
    "age": age,
    "weekly_study_hours": study_hours,
    "videos_watched": videos,
    "assignments_submitted": assignments,
    "forum_posts": forum_posts,
    "login_days": login_days,
    "previous_score": previous_score
}])

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample).max() * 100

print(f"\nPredicted Course Completion: {prediction}")
print(f"Prediction Confidence: {probability:.2f}%")
