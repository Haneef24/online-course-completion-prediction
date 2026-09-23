# Online Course Completion Prediction

## Description
This machine learning project predicts whether a student will complete an online course.

## Technology
- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Joblib
- Matplotlib

## Input Features
- Age
- Weekly study hours
- Videos watched
- Assignments submitted
- Forum posts
- Login days
- Previous score

## Files
- `train_model.py` - trains and evaluates the model
- `predict.py` - predicts course completion for a new student
- `data/online_course_completion.csv` - synthetic dataset
- `online_course_completion_model.pkl` - trained model
- `confusion_matrix.png` - evaluation chart
- `requirements.txt` - required libraries

## Run
```bash
pip install -r requirements.txt
python train_model.py
python predict.py
streamlit run app.py
```

## Note
The dataset is synthetic and intended for educational demonstration purposes.
