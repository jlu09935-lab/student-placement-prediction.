import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# ---------------------------
# Load saved model, scaler, columns
# ---------------------------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

# Columns that were scaled during training (numeric, non-binary columns)
scale_cols = ['Age', 'SSC_%', 'HSC_%', 'Degree_%', 'CGPA', 'Aptitude', 'Logical',
              'Coding', 'Communication', 'English', 'Hackathon_Score', 'Internships',
              'Projects', 'Certifications', 'Attendance_%', 'Backlogs',
              'Mock_Interview', 'Resume_Score', 'Expected_Salary_LPA']
scale_cols = [c for c in scale_cols if c in feature_columns]

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Placement Prediction System")
st.write("Enter student details below to predict placement outcome.")

st.header("Personal Details")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=18, max_value=30, value=22)
    state = st.selectbox("State", ["Delhi", "Gujarat", "Karnataka", "MP", "Maharashtra", "Rajasthan", "UP"])
with col2:
    degree_stream = st.selectbox("Degree Stream", ["BCA", "BSc", "BTech", "MBA", "MCA"])
    specialization = st.selectbox("Specialization", ["AI", "CSE", "Civil", "Data Science", "ECE", "EEE", "IT", "Mechanical"])
    college_tier = st.selectbox("College Tier", ["Tier-1", "Tier-2", "Tier-3"])

st.header("Academic Scores")
col3, col4 = st.columns(2)
with col3:
    ssc = st.slider("SSC %", 0.0, 100.0, 75.0)
    hsc = st.slider("HSC %", 0.0, 100.0, 75.0)
    degree_pct = st.slider("Degree %", 0.0, 100.0, 75.0)
with col4:
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)
    attendance = st.slider("Attendance %", 0.0, 100.0, 85.0)
    backlogs = st.number_input("Backlogs", min_value=0, max_value=20, value=0)

st.header("Skill Scores")
col5, col6 = st.columns(2)
with col5:
    aptitude = st.slider("Aptitude Score", 0, 100, 70)
    logical = st.slider("Logical Score", 0, 100, 70)
    coding = st.slider("Coding Score", 0, 100, 70)
    communication = st.slider("Communication Score", 0, 100, 70)
with col6:
    english = st.slider("English Score", 0, 100, 70)
    hackathon = st.slider("Hackathon Score", 0, 100, 50)
    mock_interview = st.slider("Mock Interview Score", 0, 100, 70)
    resume_score = st.slider("Resume Score", 0, 100, 70)

st.header("Experience & Training")
col7, col8 = st.columns(2)
with col7:
    internships = st.number_input("Internships", min_value=0, max_value=10, value=1)
    projects = st.number_input("Projects", min_value=0, max_value=20, value=2)
    certifications = st.number_input("Certifications", min_value=0, max_value=20, value=1)
with col8:
    work_experience = st.selectbox("Work Experience", ["No", "Yes"])
    leadership = st.selectbox("Leadership Role", ["No", "Yes"])
    placement_training = st.selectbox("Placement Training", ["Completed", "Not Enrolled", "Ongoing"])

expected_salary = st.number_input("Expected Salary (LPA)", min_value=0.0, max_value=100.0, value=6.0)

# ---------------------------
# Predict button
# ---------------------------
if st.button("Predict Placement"):

    # Build a single-row dict with all raw inputs
    input_dict = {
        'Gender': 1 if gender == 'Male' else 0,
        'Age': age,
        'SSC_%': ssc,
        'HSC_%': hsc,
        'Degree_%': degree_pct,
        'CGPA': cgpa,
        'Aptitude': aptitude,
        'Logical': logical,
        'Coding': coding,
        'Communication': communication,
        'English': english,
        'Hackathon_Score': hackathon,
        'Internships': internships,
        'Projects': projects,
        'Certifications': certifications,
        'Work_Experience': 1 if work_experience == 'Yes' else 0,
        'Leadership': 1 if leadership == 'Yes' else 0,
        'Attendance_%': attendance,
        'Backlogs': backlogs,
        'Mock_Interview': mock_interview,
        'Resume_Score': resume_score,
        'Expected_Salary_LPA': expected_salary,
    }

    # One-hot style flags (must match training columns exactly, drop_first base category = 0 for all)
    for s in ['Gujarat', 'Karnataka', 'MP', 'Maharashtra', 'Rajasthan', 'UP']:
        input_dict[f'State_{s}'] = 1 if state == s else 0

    for d in ['BSc', 'BTech', 'MBA', 'MCA']:
        input_dict[f'Degree_Stream_{d}'] = 1 if degree_stream == d else 0

    for sp in ['CSE', 'Civil', 'Data Science', 'ECE', 'EEE', 'IT', 'Mechanical']:
        input_dict[f'Specialization_{sp}'] = 1 if specialization == sp else 0

    for t in ['Tier-2', 'Tier-3']:
        input_dict[f'College_Tier_{t}'] = 1 if college_tier == t else 0

    for pt in ['Not Enrolled', 'Ongoing']:
        input_dict[f'Placement_Training_{pt}'] = 1 if placement_training == pt else 0

    # Build dataframe with exact column order used in training
    input_df = pd.DataFrame([input_dict])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    # Scale numeric columns using the saved scaler
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ Likely to be PLACED — Confidence: {probability[1]*100:.2f}%")
    else:
        st.error(f"❌ Likely NOT to be Placed — Confidence: {probability[0]*100:.2f}%")

    st.write("Probability breakdown:")
    st.write(f"- Not Placed: {probability[0]*100:.2f}%")
    st.write(f"- Placed: {probability[1]*100:.2f}%")
