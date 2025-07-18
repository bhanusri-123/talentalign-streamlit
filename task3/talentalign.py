import streamlit as st
import fitz
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Replacing NLTK with built-in stopword list and simple tokenizer
stop_words = set([
    'a', 'an', 'the', 'in', 'on', 'and', 'or', 'for', 'to', 'with',
    'is', 'are', 'was', 'were', 'of', 'at', 'by', 'from', 'this',
    'that', 'it', 'as', 'be', 'can', 'will', 'which', 'has', 'have'
])

def simple_tokenizer(text):
    return text.split()

# Predefined keyword mappings
PREDEFINED_SKILLS = {
    'python': ['python', 'py', 'flask', 'pandas', 'numpy'],
    'javascript': ['javascript', 'js', 'node', 'react', 'angular'],
    'sql': ['sql', 'mysql', 'postgresql', 'mongodb'],
    'html': ['html', 'css', 'bootstrap', 'tailwind'],
    'ml': ['ml', 'machinelearning', 'tensorflow', 'pytorch', 'scikit'],
    'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
}

PREDEFINED_EDUCATION = {
    'bachelors': ['btech', 'bachelor', 'b.sc', 'b.e'],
    'masters': ['mtech', 'masters', 'm.sc', 'mba'],
    'phd': ['phd', 'doctorate'],
}

PREDEFINED_EXPERIENCE = {
    'internship': ['intern', 'trainee'],
    'developer': ['developer', 'engineer'],
    'analyst': ['analyst', 'data scientist'],
    'manager': ['manager', 'lead', 'supervisor'],
}

# Functions

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def clean_text(text):
    text = text.lower()
    abbrev = {'js': 'javascript', 'ml': 'machinelearning', 'ai': 'artificialintelligence'}
    for k, v in abbrev.items():
        text = re.sub(r'\b'+re.escape(k)+r'\b', v, text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = simple_tokenizer(text)
    return ' '.join([w for w in tokens if w not in stop_words and len(w) > 2])

def extract_by_header(text, header):
    pattern = rf"(?i){header}.*?(?=\n[A-Z][^\n]*\n|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else ""


def structured_extract(raw_text):
    return {
        "skills": extract_by_header(raw_text, "skills"),
        "education": extract_by_header(raw_text, "education"),
        "experience": extract_by_header(raw_text, "experience|projects|work experience|internship")
    }

def keyword_match(text, predefined_dict):
    tokens = set(text.split())
    found = [k for k, v in predefined_dict.items() if any(word in tokens for word in v)]
    return found

def calculate_cosine(text1, text2):
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([text1, text2])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

def match_percentage(resume_items, jd_items):
    if not jd_items:
        return 0.0
    return round((len(set(resume_items) & set(jd_items)) / len(jd_items)) * 100, 2)

def section_to_keywords(text):
    return sorted(set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9.+#-]*\b', text.lower())))

# Streamlit App
st.set_page_config(page_title="TalentAlign AI", layout="wide")
st.title("🤖 TalentAlign AI: Resume + JD Matcher")

resume_file = st.file_uploader("📄 Upload Resume", type="pdf")
jd_file = st.file_uploader("📄 Upload JD", type="pdf")
threshold = st.slider("🎚 Match Threshold (%)", 0, 100, 70)
show_sections = st.checkbox("🧩 Show Section-Based Extraction")

if st.button("🔍 Analyze Match"):
    if resume_file and jd_file:
        resume_raw = extract_text_from_pdf(resume_file)
        jd_raw = extract_text_from_pdf(jd_file)

        resume_clean = clean_text(resume_raw)
        jd_clean = clean_text(jd_raw)

        r_skills = keyword_match(resume_clean, PREDEFINED_SKILLS)
        j_skills = keyword_match(jd_clean, PREDEFINED_SKILLS)
        r_edu = keyword_match(resume_clean, PREDEFINED_EDUCATION)
        j_edu = keyword_match(jd_clean, PREDEFINED_EDUCATION)
        r_exp = keyword_match(resume_clean, PREDEFINED_EXPERIENCE)
        j_exp = keyword_match(jd_clean, PREDEFINED_EXPERIENCE)

        resume_sections = structured_extract(resume_raw)
        jd_sections = structured_extract(jd_raw)

        tfidf_score = calculate_cosine(resume_clean, jd_clean)
        bert_model = SentenceTransformer("all-MiniLM-L6-v2")
        bert_resume = bert_model.encode(resume_clean, convert_to_tensor=True)
        bert_jd = bert_model.encode(jd_clean, convert_to_tensor=True)
        bert_score = round(util.cos_sim(bert_resume, bert_jd).item() * 100, 2)

        skills_match = match_percentage(r_skills, j_skills)
        edu_match = match_percentage(r_edu, j_edu)
        exp_match = match_percentage(r_exp, j_exp)

        custom_score = round(
            0.25 * tfidf_score +
            0.25 * bert_score +
            0.2 * skills_match +
            0.15 * edu_match +
            0.15 * exp_match, 2
        )

        st.subheader("📈 Similarity Breakdown")
        col1, col2, col3 = st.columns(3)
        col1.metric("TF-IDF Similarity", f"{tfidf_score}%")
        col2.metric("Sentence-BERT Similarity", f"{bert_score}%")
        col3.metric("Skills Match", f"{skills_match}%")

        col4, col5, col6 = st.columns(3)
        col4.metric("Education Match", f"{edu_match}%")
        col5.metric("Experience Match", f"{exp_match}%")
        col6.metric("Custom Score", f"{custom_score}%")

        if show_sections:
            st.subheader("✅ Skills Found in Resume and JD (Section-Based)")

            rs_skills = section_to_keywords(resume_sections["skills"]) if resume_sections["skills"] else []
            jd_skills = section_to_keywords(jd_sections["skills"]) if jd_sections["skills"] else []

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🧠 Skills Found in Resume")
                if rs_skills:
                    st.write(rs_skills)
                else:
                    st.warning("No skills section found in Resume.")

            with col2:
                st.markdown("🧠 Skills Found in JD")
                if jd_skills:
                    st.write(jd_skills)
                else:
                    st.warning("No skills section found in JD.")

        st.subheader("📋 Summary Table")
        df = pd.DataFrame({
            'Category': ['Skills', 'Education', 'Experience'],
            'Resume': [len(r_skills), len(r_edu), len(r_exp)],
            'JD': [len(j_skills), len(j_edu), len(j_exp)]
        })
        st.dataframe(df, use_container_width=True)

        st.subheader("🚨 Missing JD Skills in Resume")
        missing = list(set(j_skills) - set(r_skills))
        if missing:
            st.warning(", ".join(missing))
        else:
            st.success("All JD technical skills found in Resume!")
    else:
        st.warning("Please upload both Resume and JD PDFs.")
