import streamlit as st
import fitz  # PyMuPDF
import re
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import json
import time
from typing import List, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="TalentAlign AI - Multi Resume Matcher",
    page_icon="🎯",
    layout="wide"
)

# Initialize OpenAI client
@st.cache_resource
def init_openai_client():
    """Initialize OpenAI client with API key"""
    api_key = st.secrets.get("OPENAI_API_KEY") or st.sidebar.text_input(
        "🔑 Enter OpenAI API Key", 
        type="password",
        help="Required for AI-powered analysis"
    )
    
    if not api_key:
        st.warning("⚠ Please provide OpenAI API key to use AI features")
        return None
    
    return OpenAI(api_key=api_key)

st.title("🎯 TalentAlign AI - Multi-Resume Matcher")
st.markdown("Find the perfect candidate from multiple resumes with AI-powered ranking and experience prioritization")

# Initialize OpenAI client
client = init_openai_client()

# File uploaders
col1, col2 = st.columns([2, 1])
with col1:
    resume_files = st.file_uploader(
        "📄 Upload Multiple Resume PDFs", 
        type="pdf",
        accept_multiple_files=True,
        help="Upload multiple resumes to compare against the job description"
    )
    
with col2:
    jd_file = st.file_uploader("📋 Upload Job Description PDF", type="pdf")

# Configuration - Fixed parameters (modify in code if needed)
# ===== CONFIGURABLE PARAMETERS =====
EXPERIENCE_WEIGHT = 0.3  # Experience weight (0.1-0.6) - Higher values prioritize experience more
MATCH_THRESHOLD = 70   # Fixed threshold percentage for qualification (50-90)
TOP_RESULTS_DISPLAY = 3  # Number of top results to show in detailed view (change to 5, 10, etc. if needed)
# ===================================

st.subheader("⚙ Configuration")
col1, col2 = st.columns(2)

with col1:
    max_results = st.selectbox("📊 Max Results in Table", [5, 10, 15, 20], index=1,
                              help="Number of results to show in comparison table")
    
with col2:
    st.info(f"🎯 Fixed Settings:\n- Experience Weight: {EXPERIENCE_WEIGHT}\n- Qualification Threshold: {MATCH_THRESHOLD}%\n- Top Detailed Results: {TOP_RESULTS_DISPLAY}")

# Set fixed values
match_threshold = MATCH_THRESHOLD
experience_weight = EXPERIENCE_WEIGHT

# Debug mode checkbox - moved to top to be available throughout the code
show_debug = st.checkbox("🐛 Debug Mode")

# ----------------------------- TEXT EXTRACTION ----------------------------- #
@st.cache_data
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    text = ""
    try:
        pdf_data = pdf_file.read()
        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# ----------------------------- OPENAI FUNCTIONS ----------------------------- #
@st.cache_data
def get_openai_embeddings(text, _client_key):
    """Get embeddings using OpenAI's text-embedding-3-small model"""
    if not client:
        return None
    
    try:
        # Truncate text if too long
        if len(text) > 30000:
            text = text[:30000]
            
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Error getting embeddings: {str(e)}")
        return None

def analyze_resume_with_gpt(resume_text, jd_text, resume_filename, client):
    """Analyze single resume against job description using GPT-3.5-turbo"""
    if not client:
        return None
    
    system_prompt = """You are an expert HR analyst. Analyze the resume against the job description.
    Extract specific details about experience, skills, and qualifications. Be precise with years of experience."""
    
    user_prompt = f"""
    Analyze this resume against the job description:

    RESUME ({resume_filename}):
    {resume_text[:4000]}

    JOB DESCRIPTION:
    {jd_text[:4000]}

    Return analysis in JSON format:
    {{
        "overall_match_score": 0-100,
        "experience_years": number (total years of experience),
        "relevant_experience_years": number (years relevant to this role),
        "technical_skills_match": 0-100,
        "education_alignment": 0-100,
        "strengths": ["strength1", "strength2"],
        "gaps": ["gap1", "gap2"],
        "keywords_found": ["keyword1", "keyword2"],
        "missing_keywords": ["missing1", "missing2"],
        "candidate_name": "extracted name or 'Not found'",
        "key_skills": ["skill1", "skill2"],
        "certifications": ["cert1", "cert2"],
        "education_level": "degree level",
        "industry_experience": "relevant industries",
        "contact_info": "email or phone if visible",
        "summary": "brief candidate summary in 2-3 lines"
    }}
    
    Return only valid JSON, no markdown formatting.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up potential markdown formatting
        if "json" in content:
            content = content.split("json")[1].split("")[0].strip()
        elif "" in content:
            content = content.split("")[1].split("")[0].strip()
            
        return json.loads(content)
        
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON for {resume_filename}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error analyzing {resume_filename}: {str(e)}")
        return None

# ----------------------------- SCORING AND RANKING ----------------------------- #
def extract_jd_requirements(jd_text, client):
    """Extract experience and skill requirements from Job Description"""
    if not client:
        return None
    
    system_prompt = """You are an expert at analyzing job descriptions. Extract the specific requirements from this JD."""
    
    user_prompt = f"""
    Analyze this job description and extract the requirements:

    JOB DESCRIPTION:
    {jd_text[:4000]}

    Return requirements in JSON format:
    {{
        "min_experience_years": number (minimum years required, 0 if not specified),
        "preferred_experience_years": number (preferred years, same as min if not specified),
        "required_skills": ["skill1", "skill2"],
        "preferred_skills": ["skill1", "skill2"],
        "education_requirement": "degree level required",
        "experience_level": "entry/mid/senior/lead",
        "role_type": "technical/management/individual contributor",
        "industry_preference": "specific industry if mentioned"
    }}
    
    Return only valid JSON, no markdown formatting.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up potential markdown formatting
        if "json" in content:
            content = content.split("json")[1].split("")[0].strip()
        elif "" in content:
            content = content.split("")[1].split("")[0].strip()
            
        return json.loads(content)
        
    except Exception as e:
        st.error(f"Error extracting JD requirements: {str(e)}")
        return None

def calculate_dynamic_experience_score(candidate_exp, relevant_exp, jd_requirements):
    """Calculate experience score based on JD requirements"""
    if not jd_requirements:
        # Fallback to basic scoring if JD analysis failed
        if candidate_exp <= 2:
            return min(30, candidate_exp * 15)
        elif candidate_exp <= 5:
            return 30 + ((candidate_exp - 2) * 10)
        else:
            return min(85, 60 + ((candidate_exp - 5) * 5))
    
    min_required = jd_requirements.get('min_experience_years', 0)
    preferred = jd_requirements.get('preferred_experience_years', min_required + 2)
    
    # Dynamic scoring based on JD requirements
    if candidate_exp < min_required:
        # Below minimum - low score
        if min_required > 0:
            score = (candidate_exp / min_required) * 40  # Max 40% if below minimum
        else:
            score = min(50, candidate_exp * 10)  # If no requirement, basic scoring
    elif candidate_exp >= preferred:
        # Meets or exceeds preferred - high score
        base_score = 85
        excess_years = candidate_exp - preferred
        # Bonus for experience above preferred (diminishing returns)
        bonus = min(15, excess_years * 2)
        score = min(100, base_score + bonus)
    else:
        # Between minimum and preferred - linear scaling
        range_size = preferred - min_required
        if range_size > 0:
            progress = (candidate_exp - min_required) / range_size
            score = 40 + (progress * 45)  # Scale from 40% to 85%
        else:
            score = 85  # If min == preferred and candidate meets it
    
    # Boost for relevant experience
    if relevant_exp > 0:
        relevant_boost = min(15, relevant_exp * 2)
        score = min(100, score + relevant_boost)
    
    return round(score, 1)

def calculate_composite_score(gpt_analysis, embedding_similarity, experience_weight, jd_requirements=None):
    """Calculate composite score with dynamic experience weighting based on JD requirements"""
    if not gpt_analysis:
        return 0
    
    # Base scores (0-100 each)
    match_score = gpt_analysis.get('overall_match_score', 0)
    technical_score = gpt_analysis.get('technical_skills_match', 0)
    education_score = gpt_analysis.get('education_alignment', 0)
    
    # Dynamic experience scoring based on JD requirements
    total_exp = gpt_analysis.get('experience_years', 0)
    relevant_exp = gpt_analysis.get('relevant_experience_years', 0)
    
    exp_score = calculate_dynamic_experience_score(total_exp, relevant_exp, jd_requirements)
    
    # Fixed weighted composite score (total = 100%)
    skill_weight = 1 - experience_weight
    
    composite = (
        match_score * 0.25 * skill_weight +           # GPT match score
        technical_score * 0.20 * skill_weight +       # Technical skills
        education_score * 0.05 * skill_weight +       # Education
        embedding_similarity * 0.25 +                 # Semantic similarity (fixed 25%)
        exp_score * experience_weight                  # Experience weight (dynamic)
    )
    
    return round(composite, 1)

def process_multiple_resumes(resume_files, jd_text, client):
    """Process multiple resumes and return ranked results"""
    results = []
    
    # First, extract JD requirements
    st.text("🔍 Analyzing Job Description requirements...")
    jd_requirements = extract_jd_requirements(jd_text, client)
    
    if jd_requirements:
        st.success(f"✅ JD Analysis: Requires {jd_requirements.get('min_experience_years', 0)}-{jd_requirements.get('preferred_experience_years', 0)} years experience")
    else:
        st.warning("⚠ Could not extract JD requirements, using fallback scoring")
    
    # Get JD embedding once
    client_key = str(hash(str(client.api_key))) if hasattr(client, 'api_key') else "default"
    jd_embedding = get_openai_embeddings(jd_text, client_key)
    
    total_files = len(resume_files)
    progress_bar = st.progress(0)
    status_container = st.empty()
    
    for idx, resume_file in enumerate(resume_files):
        status_container.text(f"Processing {resume_file.name} ({idx + 1}/{total_files})")
        
        # Extract resume text
        resume_text = extract_text_from_pdf(resume_file)
        if not resume_text:
            continue
            
        # Get resume embedding
        resume_embedding = get_openai_embeddings(resume_text, client_key)
        embedding_similarity = calculate_embedding_similarity(resume_embedding, jd_embedding)
        
        # GPT Analysis
        gpt_analysis = analyze_resume_with_gpt(resume_text, jd_text, resume_file.name, client)
        
        if gpt_analysis:
            # Calculate composite score with JD requirements
            composite_score = calculate_composite_score(
                gpt_analysis, embedding_similarity, experience_weight, jd_requirements
            )
            
            result = {
                'filename': resume_file.name,
                'composite_score': composite_score,
                'embedding_similarity': embedding_similarity,
                'gpt_analysis': gpt_analysis,
                'jd_requirements': jd_requirements,
                'resume_text': resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text
            }
            results.append(result)
        
        progress_bar.progress((idx + 1) / total_files)
    
    progress_bar.empty()
    status_container.empty()
    
    # Sort by composite score
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return results

def calculate_embedding_similarity(embedding1, embedding2):
    """Calculate cosine similarity between embeddings"""
    if embedding1 is None or embedding2 is None:
        return 0
    
    try:
        similarity = cosine_similarity(
            embedding1.reshape(1, -1), 
            embedding2.reshape(1, -1)
        )[0][0]
        return round(similarity * 100, 2)
    except Exception as e:
        return 0

# ----------------------------- MAIN ANALYSIS ----------------------------- #
if st.button("🚀 Analyze All Resumes", type="primary", use_container_width=True):
    if not client:
        st.error("❌ Please provide OpenAI API key to proceed")
        st.stop()
    
    if not resume_files or not jd_file:
        st.warning("⚠ Please upload resumes and job description PDF")
        st.stop()
    
    if len(resume_files) > 20:
        st.warning("⚠ Maximum 20 resumes allowed at once")
        st.stop()
    
    # Extract JD text
    jd_text = extract_text_from_pdf(jd_file)
    if not jd_text:
        st.error("❌ Failed to extract job description text")
        st.stop()
    
    st.subheader(f"🔄 Processing {len(resume_files)} Resumes...")
    
    with st.spinner("AI is analyzing all resumes..."):
        results = process_multiple_resumes(resume_files, jd_text, client)
    
    if not results:
        st.error("❌ No valid results found")
        st.stop()
    
    # ----------------------------- RESULTS DISPLAY ----------------------------- #
    st.success(f"✅ Analysis complete! Processed {len(results)} resumes")
    
    # Summary metrics with JD requirements context
    st.subheader("📊 Summary Statistics")
    
    if results and results[0].get('jd_requirements'):
        jd_req = results[0]['jd_requirements']
        st.info(f"JD Requirements: {jd_req.get('min_experience_years', 0)}-{jd_req.get('preferred_experience_years', 'N/A')} years • {jd_req.get('experience_level', 'N/A')} level • {jd_req.get('role_type', 'N/A')}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    scores = [r['composite_score'] for r in results]
    qualified_count = len([s for s in scores if s >= match_threshold])
    
    with col1:
        st.metric("📁 Total Resumes", len(results))
    with col2:
        st.metric("✅ Qualified Candidates", qualified_count)
    with col3:
        st.metric("📈 Average Score", f"{np.mean(scores):.1f}%")
    with col4:
        st.metric("🏆 Top Score", f"{max(scores):.1f}%")
    
    # Score distribution chart - REMOVED as requested
    
    # ----------------------------- RANKED RESULTS ----------------------------- #
    st.subheader("🏆 Ranked Results")
    
    # Display only top 3 candidates in detailed view
    display_count = min(TOP_RESULTS_DISPLAY, len(results))
    
    for rank, result in enumerate(results[:display_count], 1):
        analysis = result['gpt_analysis']
        composite_score = result['composite_score']
        
        # Determine card color based on score
        if composite_score >= match_threshold:
            card_color = "🟢"
            status = "QUALIFIED"
        elif composite_score >= match_threshold - 10:
            card_color = "🟡"
            status = "POTENTIAL"
        else:
            card_color = "🔴"
            status = "BELOW THRESHOLD"
        
        with st.expander(f"#{rank} - {result['filename']} | {composite_score:.1f}% {card_color} {status}", 
                        expanded=(rank <= 3)):
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"👤 Candidate:** {analysis.get('candidate_name', 'Name not found')}")
                st.markdown(f"📧 Contact:** {analysis.get('contact_info', 'Not provided')}")
                st.markdown(f"📝 Summary:** {analysis.get('summary', 'No summary available')}")
            
            with col2:
                st.markdown("📊 Scores**")
                st.write(f"• Composite: {composite_score:.1f}%")
                st.write(f"• Match: {analysis.get('overall_match_score', 0)}%")
                st.write(f"• Technical: {analysis.get('technical_skills_match', 0)}%")
                st.write(f"• Embedding: {result['embedding_similarity']}%")
                
            with col3:
                st.markdown("💼 Experience vs Requirements**")
                total_exp = analysis.get('experience_years', 0)
                relevant_exp = analysis.get('relevant_experience_years', 0)
                
                # Show experience vs JD requirements
                if result.get('jd_requirements'):
                    jd_req = result['jd_requirements']
                    min_req = jd_req.get('min_experience_years', 0)
                    pref_req = jd_req.get('preferred_experience_years', min_req)
                    
                    exp_status = "✅ Exceeds" if total_exp >= pref_req else "✅ Meets" if total_exp >= min_req else "❌ Below"
                    st.write(f"• Total: {total_exp} years ({exp_status})")
                    st.write(f"• Relevant: {relevant_exp} years")
                    st.write(f"• Required: {min_req}-{pref_req} years")
                else:
                    st.write(f"• Total: {total_exp} years")
                    st.write(f"• Relevant: {relevant_exp} years")
                
                st.write(f"• Education: {analysis.get('education_level', 'Not specified')}")
            
            # Detailed breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("✅ Strengths:")
                for strength in analysis.get('strengths', [])[:3]:
                    st.write(f"• {strength}")
                
                st.markdown("🔧 Key Skills:")
                for skill in analysis.get('key_skills', [])[:5]:
                    st.write(f"• {skill}")
            
            with col2:
                st.markdown("❌ Gaps:")
                for gap in analysis.get('gaps', [])[:3]:
                    st.write(f"• {gap}")
                
                st.markdown("🎓 Certifications:")
                certs = analysis.get('certifications', [])
                if certs:
                    for cert in certs[:3]:
                        st.write(f"• {cert}")
                else:
                    st.write("• None mentioned")
            
            # Keywords analysis
            col1, col2 = st.columns(2)
            with col1:
                found_keywords = analysis.get('keywords_found', [])
                if found_keywords:
                    st.markdown("🎯 Matched Keywords:")
                    st.write(", ".join(found_keywords[:8]))
            
            with col2:
                missing_keywords = analysis.get('missing_keywords', [])
                if missing_keywords:
                    st.markdown("❌ Missing Keywords:")
                    st.write(", ".join(missing_keywords[:6]))
    
    # ----------------------------- COMPARISON TABLE ----------------------------- #
    st.subheader("📋 Quick Comparison Table")
    
    # Create comparison dataframe
    comparison_data = []
    for rank, result in enumerate(results[:max_results], 1):
        analysis = result['gpt_analysis']
        comparison_data.append({
            'Rank': rank,
            'Filename': result['filename'][:30] + "..." if len(result['filename']) > 30 else result['filename'],
            'Composite Score': f"{result['composite_score']:.1f}%",
            'Experience (Years)': analysis.get('experience_years', 0),
            'Relevant Exp': analysis.get('relevant_experience_years', 0),
            'Technical Match': f"{analysis.get('technical_skills_match', 0)}%",
            'Education': analysis.get('education_level', 'N/A'),
            'Status': '🟢 Qualified' if result['composite_score'] >= match_threshold 
                     else '🟡 Potential' if result['composite_score'] >= match_threshold - 10
                     else '🔴 Below Threshold'
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Export functionality
    st.subheader("📤 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Create detailed export data
        export_data = []
        for result in results:
            analysis = result['gpt_analysis']
            export_data.append({
                'filename': result['filename'],
                'composite_score': result['composite_score'],
                'embedding_similarity': result['embedding_similarity'],
                'candidate_name': analysis.get('candidate_name', ''),
                'contact_info': analysis.get('contact_info', ''),
                'experience_years': analysis.get('experience_years', 0),
                'relevant_experience_years': analysis.get('relevant_experience_years', 0),
                'overall_match_score': analysis.get('overall_match_score', 0),
                'technical_skills_match': analysis.get('technical_skills_match', 0),
                'education_alignment': analysis.get('education_alignment', 0),
                'key_skills': ', '.join(analysis.get('key_skills', [])),
                'strengths': ', '.join(analysis.get('strengths', [])),
                'gaps': ', '.join(analysis.get('gaps', [])),
                'summary': analysis.get('summary', '')
            })
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Detailed Results (CSV)",
            data=csv,
            file_name=f"resume_analysis_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Quick summary for HR
        summary_text = f"""# Resume Analysis Summary - {time.strftime('%Y-%m-%d %H:%M')}

## Analysis Parameters
- Total Resumes Analyzed: {len(results)}
- Match Threshold: {match_threshold}%
- Experience Weight: {experience_weight}

## Top 5 Candidates:
"""
        
        for i, result in enumerate(results[:5], 1):
            analysis = result['gpt_analysis']
            summary_text += f"""
{i}. {result['filename']} - {result['composite_score']:.1f}%
   - Candidate: {analysis.get('candidate_name', 'N/A')}
   - Experience: {analysis.get('experience_years', 0)} years total, {analysis.get('relevant_experience_years', 0)} relevant
   - Contact: {analysis.get('contact_info', 'N/A')}
"""
        
        st.download_button(
            label="📝 Download Summary Report",
            data=summary_text,
            file_name=f"resume_summary_{time.strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

# ----------------------------- SIDEBAR INFO ----------------------------- #
with st.sidebar:
    st.header("📖 How It Works")
    st.markdown("""
    Multi-Resume Analysis Process:
    
    1. 📤 Upload multiple resumes + 1 JD
    2. 🤖 AI extracts candidate details
    3. 📊 Calculates composite scores:
       - Skills match
       - Experience weighting
       - Semantic similarity
       - Leadership bonus
    4. 🏆 Ranks candidates automatically
    5. 📋 Provides detailed comparison
    
    Scoring Components:
    - Skills Match: Technical + soft skills (25-45%)
    - Experience: Dynamic scoring vs JD requirements (10-60%)
    - Education: Degree alignment (2-4.5%)
    - Semantic: AI context understanding (25%)
    
    Dynamic Experience Scoring:
    - Extracts required years from JD automatically
    - Below minimum: 0-40% (penalty for not meeting requirements)
    - Minimum to preferred: 40-85% (linear scaling)
    - Above preferred: 85-100% (bonus for exceeding)
    - Relevant experience: +2 points/year bonus (max +15)
    """)
    
    if show_debug and 'results' in locals():
        st.header("🐛 Debug Info")
        st.write(f"Processed: {len(results)} resumes")
        st.write(f"Experience weight: {experience_weight}")
        st.write(f"Fixed threshold: {match_threshold}%")

# Footer
st.markdown("---")
st.markdown("TalentAlign AI Multi-Resume Matcher - Powered by OpenAI GPT-3.5-turbo")
