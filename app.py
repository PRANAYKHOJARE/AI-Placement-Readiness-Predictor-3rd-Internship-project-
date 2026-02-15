import streamlit as st
import pickle
import pandas as pd
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import tempfile

# Page Config
st.set_page_config(
    page_title="AI Placement Predictor",
    page_icon="🤖",
    layout="wide"
)

# Load model
model, feature_columns = pickle.load(open("placement_model.pkl", "rb"))

# Custom CSS Styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 18px;
        }
        .metric-box {
            padding: 20px;
            background-color: white;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 AI-Based Student Placement Readiness Predictor")
st.markdown("### Smart AI System to Evaluate Job Readiness")

# Sidebar Input
st.sidebar.header("📌 Enter Student Details")

cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)
internships = st.sidebar.slider("Internships Completed", 0, 5, 1)
coding = st.sidebar.slider("Coding Skill Rating", 1, 5, 3)
communication = st.sidebar.slider("Communication Skill Rating", 1, 5, 3)
aptitude = st.sidebar.slider("Aptitude Skill Rating", 1, 5, 3)
projects = st.sidebar.slider("Projects Completed", 0, 10, 2)

predict_btn = st.sidebar.button("🎯 Predict Readiness")

if predict_btn:

    input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    if "cgpa" in input_df.columns:
        input_df["cgpa"] = cgpa
    if "internships_completed" in input_df.columns:
        input_df["internships_completed"] = internships
    if "coding_skill_rating" in input_df.columns:
        input_df["coding_skill_rating"] = coding
    if "communication_skill_rating" in input_df.columns:
        input_df["communication_skill_rating"] = communication
    if "aptitude_skill_rating" in input_df.columns:
        input_df["aptitude_skill_rating"] = aptitude
    if "projects_completed" in input_df.columns:
        input_df["projects_completed"] = projects

    score = model.predict(input_df)[0]
    percentage = min((score / 300) * 100, 100)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Placement Readiness Score")
        st.progress(int(percentage))
        st.metric("Readiness Percentage", f"{percentage:.2f}%")

    with col2:
        if percentage >= 75:
            result_text = "🔥 High Placement Readiness"
            st.success(result_text)
        elif percentage >= 50:
            result_text = "⚠ Moderate Placement Readiness"
            st.warning(result_text)
        else:
            result_text = "❌ Low Placement Readiness"
            st.error(result_text)

    # Suggestions
    st.markdown("## 💡 Skill Improvement Suggestions")
    suggestions = []

    if cgpa < 7:
        suggestions.append("📘 Improve academic performance (Target CGPA > 7.5)")
    if internships < 2:
        suggestions.append("🏢 Gain at least 2 internships")
    if coding < 4:
        suggestions.append("💻 Practice DSA & coding regularly")
    if communication < 4:
        suggestions.append("🗣 Improve communication & mock interviews")
    if aptitude < 4:
        suggestions.append("🧠 Practice aptitude questions daily")
    if projects < 3:
        suggestions.append("🚀 Build 2–3 real-world projects")

    if suggestions:
        for s in suggestions:
            st.write(s)
    else:
        st.success("Excellent Profile! Keep Growing 🚀")

    # -------- PDF GENERATION --------
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_file.name, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>AI Placement Readiness Report</b>", styles["Title"]))
    elements.append(Spacer(1, 20))

    data = [
        ["CGPA", cgpa],
        ["Internships", internships],
        ["Coding Skill", coding],
        ["Communication Skill", communication],
        ["Aptitude Skill", aptitude],
        ["Projects", projects],
        ["Readiness Score (%)", f"{percentage:.2f}%"],
        ["Result", result_text],
    ]

    table = Table(data)
    elements.append(table)
    elements.append(Spacer(1, 20))

    for s in suggestions:
        elements.append(Paragraph("- " + s, styles["Normal"]))

    doc.build(elements)

    with open(temp_file.name, "rb") as f:
        st.download_button(
            label="📥 Download Detailed Report",
            data=f,
            file_name="Placement_Readiness_Report.pdf",
            mime="application/pdf"
        )
