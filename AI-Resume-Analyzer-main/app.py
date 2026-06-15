import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from resume_parser import extract_text_from_pdf

st.title("🤖 AI Resume Analyzer")
st.write("Upload your resume and compare with job description")

# Skill list
skills_list = [
    "python",
    "machine learning",
    "data analysis",
    "sql",
    "deep learning",
    "power bi",
    "tableau",
    "excel",
    "java",
    "c++",
    "html",
    "css",
    "javascript"
]

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

job_description = st.text_area("Paste Job Description")

if uploaded_file is not None and job_description:

    resume_text = extract_text_from_pdf(uploaded_file)

    documents = [resume_text, job_description]

    # TF-IDF
    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    score = similarity[0][0] * 100

    st.subheader("📊 Resume Match Score")

    st.success(f"{score:.2f}% Match")

    # Skill extraction
    resume_text_lower = resume_text.lower()

    found_skills = []
    missing_skills = []

    for skill in skills_list:

        if skill in resume_text_lower:
            found_skills.append(skill)
        else:
            missing_skills.append(skill)

    st.subheader("✅ Skills Found in Resume")

    if found_skills:
        for skill in found_skills:
            st.write("✔", skill)
    else:
        st.write("No skills detected")

    st.subheader("❌ Suggested Skills to Add")

    for skill in missing_skills[:5]:
        st.write("➜", skill)