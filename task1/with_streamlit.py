import streamlit as st
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# UI Title
st.title("📄 Upload Resume and Job Description")

# Debug log: App started
st.write("App started successfully.")

# File uploaders
resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"])

# Debug log: File status
st.write("Resume uploaded:", resume_file is not None)
st.write("JD uploaded:", jd_file is not None)

if resume_file and jd_file:
    st.info("📂 Both files uploaded. Starting extraction...")

    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)

    # Save extracted text to .txt files
    with open("resume_text.txt", "w", encoding="utf-8") as f:
        f.write(resume_text)
    with open("jd_text.txt", "w", encoding="utf-8") as f:
        f.write(jd_text)

    st.subheader("📄 Resume Text")
    st.text_area("Resume", resume_text, height=200)

    st.subheader("📋 Job Description Text")
    st.text_area("JD", jd_text, height=200)

    st.success("✅ Text saved to resume_text.txt and jd_text.txt")
else:
    st.warning("📤 Please upload both PDF files to continue.")
