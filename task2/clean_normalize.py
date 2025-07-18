import streamlit as st
import fitz  # PyMuPDF
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()

# ------------------ PDF TEXT EXTRACTION ------------------ #
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# ------------------ TEXT CLEANING ------------------ #
def clean_text(text):
    text = text.lower()
    abbreviations = {
        'js': 'javascript',
        'ml': 'machinelearning',
        'ai': 'artificialintelligence',
        'c++': 'cplusplus',
        'c#': 'csharp',
        '.net': 'dotnet'
    }
    for key, value in abbreviations.items():
        text = re.sub(r'\b' + re.escape(key) + r'\b', value, text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    tokens = tokenizer.tokenize(text)
    filtered_words = [word for word in tokens if word not in stop_words and len(word) > 2]

    return ' '.join(filtered_words)

# ------------------ STREAMLIT UI ------------------ #
st.set_page_config(page_title="Upload + Clean", layout="wide")
st.title("📄 Upload & 🧹 Clean Resume and JD Text")

resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"])

st.write("Resume uploaded:", resume_file is not None)
st.write("JD uploaded:", jd_file is not None)

if resume_file and jd_file:
    st.info("📂 Both files uploaded. Extracting text...")

    # Extract raw text
    resume_raw = extract_text_from_pdf(resume_file)
    jd_raw = extract_text_from_pdf(jd_file)

    st.subheader("📄 Resume Text (Raw)")
    st.text_area("Resume Raw", resume_raw, height=150)

    st.subheader("📋 JD Text (Raw)")
    st.text_area("JD Raw", jd_raw, height=150)

    # Clean the text
    resume_clean = clean_text(resume_raw)
    jd_clean = clean_text(jd_raw)

    # Show cleaned text
    st.subheader("🧼 Cleaned Resume Text")
    st.text_area("Resume Cleaned", resume_clean, height=150)

    st.subheader("🧼 Cleaned JD Text")
    st.text_area("JD Cleaned", jd_clean, height=150)

    # Save files
    with open("resume_text.txt", "w", encoding="utf-8") as f:
        f.write(resume_raw)
    with open("jd_text.txt", "w", encoding="utf-8") as f:
        f.write(jd_raw)
    with open("resume_cleaned.txt", "w", encoding="utf-8") as f:
        f.write(resume_clean)
    with open("jd_cleaned.txt", "w", encoding="utf-8") as f:
        f.write(jd_clean)

    st.success("✅ Text extracted, cleaned, and saved to .txt files!")
else:
    st.warning("📤 Please upload both PDF files to continue.")
