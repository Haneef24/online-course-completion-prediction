import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Online Course Completion Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom EdTech Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #064E3B 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.35);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        margin-bottom: 10px;
    }

    .completion-card-yes {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(5, 150, 105, 0.06) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .completion-card-no {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(185, 28, 28, 0.06) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .stat-hero {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 6px 0;
    }

    .pedagogy-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #10B981;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find assets
def get_asset_path(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        script_dir / filename,
        script_dir / "Online_Course_Completion_Prediction_Sklearn" / filename,
        script_dir.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename

@st.cache_resource
def load_model():
    model_path = get_asset_path("online_course_completion_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">EdTech Learning Analytics & Student Retention AI</div>
    <h1 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 2.2rem;">🎓 Online Course Completion Predictor</h1>
    <p style="color: #94A3B8; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Forecast student completion and dropout risk using study hours, assignment submissions, video lecture completion, and active platform login days.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🎯 Learner Retention Forecast", "📁 Cohort Retention Screening (CSV)", "📊 Model Performance & Confusion Matrix"])

# --- TAB 1: Learner Retention Forecast ---
with tabs[0]:
    st.subheader("Student Learning Habits & Academic Parameters")

    # Quick Presets
    p_cols = st.columns([1, 1, 1, 3])
    with p_cols[0]:
        load_pass = st.button("🌟 Dedicated Student (High Completion)", width="stretch")
    with p_cols[1]:
        load_risk = st.button("⚠️ At-Risk Candidate (Dropout Risk)", width="stretch")
    with p_cols[2]:
        load_sample = st.button("📋 Sample Student", width="stretch")

    if load_pass:
        st.session_state["age"] = 32
        st.session_state["hours"] = 18.0
        st.session_state["videos"] = 75
        st.session_state["assignments"] = 17
        st.session_state["posts"] = 14
        st.session_state["logins"] = 72
        st.session_state["score"] = 86.0
    elif load_risk:
        st.session_state["age"] = 24
        st.session_state["hours"] = 3.5
        st.session_state["videos"] = 12
        st.session_state["assignments"] = 3
        st.session_state["posts"] = 1
        st.session_state["logins"] = 8
        st.session_state["score"] = 42.0
    elif load_sample:
        st.session_state["age"] = 28
        st.session_state["hours"] = 10.5
        st.session_state["videos"] = 45
        st.session_state["assignments"] = 11
        st.session_state["posts"] = 6
        st.session_state["logins"] = 45
        st.session_state["score"] = 68.0

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("#### 🕒 Study Consistency & Engagement")
        study_hours = st.slider(
            "Weekly Study Hours", 1.0, 40.0,
            value=float(st.session_state.get("hours", 10.5)), step=0.5,
            key="input_hours", help="Total self-paced and guided study time logged weekly."
        )
        videos_watched = st.slider(
            "Video Lectures Completed", 0, 100,
            value=int(st.session_state.get("videos", 45)), step=1,
            key="input_videos"
        )
        login_days = st.slider(
            "Platform Active Login Days (over 90-day term)", 1, 90,
            value=int(st.session_state.get("logins", 45)), step=1,
            key="input_logins", help="Distinct calendar days student engaged with LMS."
        )
        forum_posts = st.slider(
            "Discussion Forum Posts & Replies", 0, 50,
            value=int(st.session_state.get("posts", 6)), step=1,
            key="input_posts"
        )

    with c_right:
        st.markdown("#### 📝 Academic Performance & Deliverables")
        assignments = st.slider(
            "Assignments Submitted (out of 20)", 0, 20,
            value=int(st.session_state.get("assignments", 11)), step=1,
            key="input_assignments", help="Core homework/project milestones turned in."
        )
        prev_score = st.slider(
            "Previous Assessment Score (0 - 100)", 30.0, 100.0,
            value=float(st.session_state.get("score", 68.0)), step=0.5,
            key="input_score", help="Score achieved in prior quizzes or prerequisite course."
        )
        age = st.slider(
            "Student Age", 18, 65,
            value=int(st.session_state.get("age", 28)), step=1,
            key="input_age"
        )

    st.markdown("---")
    eval_btn = st.button("🚀 Forecast Course Completion Likelihood", type="primary", width="stretch")

    student_df = pd.DataFrame([{
        "age": age,
        "weekly_study_hours": study_hours,
        "videos_watched": videos_watched,
        "assignments_submitted": assignments,
        "forum_posts": forum_posts,
        "login_days": login_days,
        "previous_score": prev_score
    }])

    pred = model.predict(student_df)[0]
    probs = model.predict_proba(student_df)[0]
    classes = list(model.classes_)
    conf = max(probs) * 100

    yes_idx = classes.index("Yes") if "Yes" in classes else 1
    yes_prob = probs[yes_idx] * 100

    st.markdown("### 📋 Completion Forecast Outcome")
    r1, r2 = st.columns([1.3, 1.7], gap="medium")

    with r1:
        if pred == "Yes":
            st.markdown(f"""
            <div class="completion-card-yes">
                <span style="font-size: 2.8rem;">🎉</span>
                <div style="color: #34D399; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 1px;">
                    Likely to Complete Course
                </div>
                <div class="stat-hero" style="color: #10B981;">
                    {conf:.1f}%
                </div>
                <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Model Prediction Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="completion-card-no">
                <span style="font-size: 2.8rem;">⚠️</span>
                <div style="color: #F87171; font-weight: 700; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 1px;">
                    At-Risk of Incompletion
                </div>
                <div class="stat-hero" style="color: #EF4444;">
                    {conf:.1f}%
                </div>
                <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Model Prediction Confidence</p>
            </div>
            """, unsafe_allow_html=True)

        st.write(f"**Estimated Graduation Probability:** {yes_prob:.1f}%")
        st.progress(float(yes_prob / 100.0))

    with r2:
        st.markdown("#### 🔍 Persistence Drivers & Risk Alerts")
        alerts = []
        if assignments >= 14:
            alerts.append(("Consistent Milestone Submissions", f"{assignments}/20 assignments turned in keeps learner on pace.", "good"))
        elif assignments < 8:
            alerts.append(("Critical Assignment Backlog", f"Only {assignments} assignments submitted; severe risk of falling behind.", "bad"))

        if login_days >= 50:
            alerts.append(("High Platform Habituation", f"{login_days} active days demonstrates sustained study routine.", "good"))
        elif login_days < 20:
            alerts.append(("Infrequent LMS Access", f"Only {login_days} active login days indicates disengagement.", "bad"))

        if study_hours >= 12.0:
            alerts.append(("Sufficient Weekly Time Dedication", f"{study_hours} hrs/week aligns with recommended curriculum tempo.", "good"))
        elif study_hours < 5.0:
            alerts.append(("Insufficient Study Allocation", f"{study_hours} hrs/week is below the minimal threshold for mastery.", "bad"))

        if alerts:
            for title, desc, kind in alerts:
                if kind == "good":
                    st.success(f"**{title}**: {desc}")
                else:
                    st.warning(f"**{title}**: {desc}")
        else:
            st.info("Student metrics are aligned with average cohort progress.")

        st.markdown(f"""
        <div class="pedagogy-box">
            <strong style="color: #34D399;">Academic Mentorship Intervention:</strong><br>
            <span style="color: #CBD5E1; font-size: 0.92rem;">
                {'Encourage student to participate in peer mentoring or explore advanced capstone projects.' if pred == 'Yes' else 'Trigger automated nudge reminder, offer assignment extension buffer, and schedule a 15-minute TA tutoring session.'}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 View Raw Student Vector"):
        st.dataframe(student_df, width="stretch")

# --- TAB 2: Cohort Retention Screening ---
with tabs[1]:
    st.subheader("Batch Student Cohort Retention Screening")
    st.write("Upload enrollment rosters or analyze against the 600-student baseline course dataset.")

    csv_file = st.file_uploader("Upload Roster CSV", type=["csv"], key="course_csv")
    df_cohort = None

    if csv_file is not None:
        df_cohort = pd.read_csv(csv_file)
        st.info(f"Loaded {len(df_cohort)} student records from file.")
    else:
        sample_path = get_asset_path("data/online_course_completion.csv")
        if os.path.exists(sample_path):
            if st.checkbox("Load baseline student cohort (`data/online_course_completion.csv`)", value=True):
                df_cohort = pd.read_csv(sample_path)
                st.info(f"Loaded {len(df_cohort)} records from baseline cohort dataset.")

    if df_cohort is not None:
        req_cols = ["age", "weekly_study_hours", "videos_watched", "assignments_submitted", "forum_posts", "login_days", "previous_score"]
        missing = [c for c in req_cols if c not in df_cohort.columns]
        if missing:
            st.error(f"Missing required columns in dataset: {missing}")
        else:
            if st.button("⚡ Run Cohort Completion Analysis", type="primary"):
                with st.spinner("Classifying cohort trajectories..."):
                    preds = model.predict(df_cohort[req_cols])
                    probs = model.predict_proba(df_cohort[req_cols])
                    y_idx = list(model.classes_).index("Yes") if "Yes" in list(model.classes_) else 1
                    grad_probs = probs[:, y_idx] * 100

                    res_df = df_cohort.copy()
                    res_df["Predicted_Completion"] = preds
                    res_df["Completion_Probability_%"] = np.round(grad_probs, 1)

                    yes_count = sum(preds == "Yes")
                    no_count = sum(preds == "No")
                    grad_rate = (yes_count / len(res_df)) * 100

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Enrolled Students", len(res_df))
                    m2.metric("Projected Completions", yes_count, delta=f"{grad_rate:.1f}%")
                    m3.metric("At-Risk Dropouts", no_count)
                    m4.metric("Average Completion Prob", f"{np.mean(grad_probs):.1f}%")

                    f_choice = st.radio("Filter Cohort View:", ["All Students", "Likely to Complete Only", "At-Risk Dropouts Only"], horizontal=True)
                    if f_choice == "Likely to Complete Only":
                        view = res_df[res_df["Predicted_Completion"] == "Yes"]
                    elif f_choice == "At-Risk Dropouts Only":
                        view = res_df[res_df["Predicted_Completion"] == "No"]
                    else:
                        view = res_df

                    st.dataframe(view, width="stretch")

                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Retention Report as CSV",
                        data=csv_export,
                        file_name="course_completion_retention_report.csv",
                        mime="text/csv"
                    )

# --- TAB 3: Model Diagnostics ---
with tabs[2]:
    st.subheader("Model Architecture & Classification Benchmark")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🤖 Retention Model Specifications
        - **Algorithm**: `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)`
        - **Monitored Dimensions**:
            - Effort: `weekly_study_hours`, `login_days`
            - Output: `assignments_submitted`, `videos_watched`, `forum_posts`
            - Baseline: `previous_score`, `age`
        - **Accuracy Benchmark**:
            - **Overall Accuracy**: **87.50%** on Stratified Test Split
            - **Precision (Completers)**: ~0.91
            - **Recall (Completers)**: ~0.90
        """)

    with c2:
        img_path = get_asset_path("confusion_matrix.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Confusion Matrix on Test Split", width="stretch")
        else:
            st.info("Confusion matrix image not found.")

st.caption("EdTech Student Retention & Learning Analytics • Scikit-learn & Streamlit")
