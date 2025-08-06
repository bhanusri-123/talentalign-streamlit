import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from openai import OpenAI
import uuid

# === CONFIG ===
COLLECTION_NAME = "resume-jd-matching"
TOP_N = 5

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text

def create_qdrant_collection(client):
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

def clean_text(text):
    return " ".join(text.split())

# === SETUP ===
st.set_page_config(page_title="TalentAlign AI – JD Matcher", layout="wide")
st.markdown("""
    <style>
        .main > div {
            padding-top: 2rem;
        }
        .uploaded-resume {
            font-weight: bold;
            color: #2c3e50;
        }
        .stButton > button {
            background-color: #0066cc;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            margin-top: 10px;
        }
        .match-box {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            border: 1px solid #e1e4e8;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        .section-header {
            font-size: 24px;
            color: #333;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 TalentAlign AI – Resume-JD Matcher")

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
qdrant = QdrantClient(url=st.secrets["QDRANT_URL"], api_key=st.secrets["QDRANT_API_KEY"])
create_qdrant_collection(qdrant)
model = SentenceTransformer("all-MiniLM-L6-v2")

# === JD Upload ===
st.markdown("<div class='section-header'>📤 Upload Job Descriptions (JDs)</div>", unsafe_allow_html=True)
jd_files = st.file_uploader("Upload multiple JD PDFs", type=["pdf", "docx"], accept_multiple_files=True)

jd_uploaded = False
if jd_files:
    if st.button("📥 Upload JDs to Qdrant"):
        for file in jd_files:
            jd_text = clean_text(extract_text_from_pdf(file))
            embedding = model.encode(jd_text)
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload={"jd_text": jd_text, "jd_name": file.name},
                    )
                ],
            )
        jd_uploaded = True
        st.success("✅ JDs uploaded and stored in Qdrant successfully!")

st.divider()

# === Resume Upload ===
st.markdown("<div class='section-header'>📄 Upload Your Resume</div>", unsafe_allow_html=True)
resume_file = st.file_uploader("Upload your resume (PDF only)", type="pdf")

if resume_file:
    resume_text = clean_text(extract_text_from_pdf(resume_file))
    st.markdown(f"<div class='uploaded-resume'>Uploaded Resume: {resume_file.name}</div>", unsafe_allow_html=True)

    if st.button("🔍 Find Top Matching JDs"):
        with st.spinner("Matching your resume with job descriptions..."):
            resume_embedding = model.encode(resume_text)
            search_results = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=resume_embedding.tolist(),
                limit=20,
            )

            seen_texts = set()
            unique_results = []
            for result in search_results:
                jd_text = result.payload.get("jd_text", "")
                if jd_text not in seen_texts:
                    seen_texts.add(jd_text)
                    unique_results.append(result)

            top_matches = unique_results[:TOP_N]

        if top_matches:
            with st.expander("📌 Top JD Matches Based on Resume"):
                for i, result in enumerate(top_matches, start=1):
                    jd_name = result.payload.get("jd_name", f"JD {i}")
                    jd_text = result.payload.get("jd_text", "[No JD Text Found]")
                    score = result.score

                    st.markdown(f"""
                    <div class="match-box">
                        <h4>{i}. {jd_name}</h4>
                        <p><strong>Similarity Score:</strong> {score:.4f}</p>
                        <p><strong>Description:</strong> {jd_text[:500]}{'...' if len(jd_text) > 500 else ''}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No matching job descriptions found.")