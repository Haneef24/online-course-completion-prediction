# online-course-completion-prediction
Machine learning project that predicts whether a student will complete an online course using engagement, study habits, assignment activity, and previous academic performance. Includes a Random Forest model, CLI prediction tool, confusion matrix, and interactive Streamlit dashboard.
#  Online Course Completion Prediction

A machine learning project that predicts whether a student is likely to complete an online course based on their learning behavior, engagement, and previous academic performance.

The project uses a **Random Forest Classifier** built with Scikit-learn and includes an interactive **Streamlit dashboard** for individual predictions and cohort-level retention analysis.

---

##  Project Overview

Online learning platforms generate a large amount of learner activity data. Understanding these patterns can help identify students who may need additional academic support.

This project analyzes student information such as:

* Weekly study hours
* Videos watched
* Assignments submitted
* Discussion forum activity
* Platform login frequency
* Previous assessment score
* Age

The trained machine learning model predicts whether the student will complete the course and provides a prediction probability.

---

##  Features

###  Individual Student Prediction

Enter a student's learning and academic information to predict their course completion outcome.

###  Completion Probability

The application displays the estimated probability that a student will complete the course.

###  Retention & Risk Alerts

The Streamlit dashboard highlights engagement patterns such as:

* Low platform activity
* Assignment backlog
* Low weekly study time
* Consistent academic participation

###  Cohort Retention Screening

Upload a CSV containing multiple students and analyze course completion predictions across the cohort.

###  Model Evaluation

The project includes:

* Accuracy evaluation
* Classification report
* Confusion matrix visualization

### \ Command-Line Prediction

A standalone `predict.py` script allows predictions directly from the terminal.

---

##  Machine Learning

The project uses a **Random Forest Classifier** from Scikit-learn.

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
```

The dataset is divided into training and testing sets using an **80/20 split** with stratification.

---

##  Input Features

| Feature                 | Description                          |
| ----------------------- | ------------------------------------ |
| `age`                   | Student age                          |
| `weekly_study_hours`    | Average weekly study time            |
| `videos_watched`        | Number of course videos watched      |
| `assignments_submitted` | Number of assignments submitted      |
| `forum_posts`           | Discussion forum posts/replies       |
| `login_days`            | Number of active platform login days |
| `previous_score`        | Previous assessment score            |

### Target

```text
course_completed
```

Possible values:

```text
Yes
No
```

---

##  Dataset

The project includes a synthetic dataset containing **600 student records**.

The dataset contains:

* `student_id`
* `age`
* `weekly_study_hours`
* `videos_watched`
* `assignments_submitted`
* `forum_posts`
* `login_days`
* `previous_score`
* `course_completed`

> **Note:** The dataset is synthetic and intended for educational and demonstration purposes. Model results should not be interpreted as validated predictions about real students.

---

##  Project Structure

```text
Online_Course_Completion_Prediction_Sklearn/
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── online_course_completion_model.pkl
├── confusion_matrix.png
│
└── data/
    └── online_course_completion.csv
```

---

##  Technologies Used

* **Python**
* **Pandas** – Data manipulation
* **Scikit-learn** – Machine learning
* **Random Forest** – Classification model
* **Joblib** – Model serialization
* **Matplotlib** – Model evaluation visualization
* **Streamlit** – Interactive web application

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Online_Course_Completion_Prediction_Sklearn.git
```

Navigate to the project directory:

```bash
cd Online_Course_Completion_Prediction_Sklearn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

##  Train the Model

To train the Random Forest model:

```bash
python train_model.py
```

This will:

1. Load the course completion dataset.
2. Separate features and target values.
3. Split the data into training and testing sets.
4. Train the Random Forest classifier.
5. Evaluate the model.
6. Save the trained model as:

```text
online_course_completion_model.pkl
```

It also generates:

```text
confusion_matrix.png
```

---

##  Make a Prediction

Run the command-line prediction script:

```bash
python predict.py
```

You will be asked to enter:

```text
Age
Weekly study hours
Videos watched
Assignments submitted
Forum posts
Login days
Previous score
```

The application then displays the predicted course completion result and prediction confidence.

---

##  Run the Streamlit Application

Launch the interactive dashboard with:

```bash
streamlit run app.py
```

The dashboard provides three main sections:

###  Learner Retention Forecast

Enter individual student information and receive a course completion prediction.

###  Cohort Retention Screening

Upload a CSV file containing student records and analyze completion predictions across multiple learners.

###  Model Performance

View the model's evaluation information and confusion matrix.

---

##  Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The evaluation output is generated automatically when running:

```bash
python train_model.py
```

---

##  Use Cases

This project demonstrates how machine learning can be applied to:

* Learning analytics
* Student engagement analysis
* Course completion prediction
* Educational data mining
* Retention monitoring
* Early academic intervention systems

---

##  Disclaimer

This project is intended for **educational and demonstration purposes**.

The dataset is synthetic, and predictions should not be used as the sole basis for decisions about real students. In real-world educational settings, models should be validated carefully, monitored for bias, and used alongside appropriate human review.

---

##  Future Improvements

Possible improvements include:

* Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
* Feature importance visualization
* Cross-validation
* ROC-AUC and Precision-Recall curves
* Model comparison with Logistic Regression, XGBoost, and Gradient Boosting
* Explainable AI using SHAP
* Database integration
* Cloud deployment
* Authentication for the Streamlit dashboard
* Automated model retraining

---

##  Author

**Haneef Sk**

If you found this project useful, consider giving the repository a ⭐ on GitHub.
