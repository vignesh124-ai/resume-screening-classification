import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.predictor import ResumePredictor
from src.preprocessor import extract_text_from_uploaded_file, clean_resume
from src.skills_dictionary import CATEGORY_SKILLS

# Page configuration
st.set_page_config(
    page_title="AI Resume Screener & Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 15px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #DBEAFE;
        color: #1D4ED8;
        margin: 3px;
    }
    .prediction-box {
        background-color: #EFF6FF;
        border: 2px solid #3B82F6;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return ResumePredictor(models_dir="models")

@st.cache_data
def load_sample_resumes():
    with open("data/sample_resumes.json", "r") as f:
        return json.load(f)

# Sidebar
predictor = load_predictor()
samples = load_sample_resumes()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/resume.png", width=70)
    st.title("Settings & Controls")
    
    st.markdown("### 🤖 Model Selection")
    model_choice = st.selectbox(
        "Choose Classification Model:",
        options=["CalibratedLinearSVC", "MultinomialNB", "LogisticRegression"],
        format_func=lambda x: {
            "CalibratedLinearSVC": "Calibrated LinearSVC (Enhanced ~70% Acc)",
            "MultinomialNB": "Multinomial Naive Bayes (Baseline ~56% Acc)",
            "LogisticRegression": "Logistic Regression (~65% Acc)"
        }.get(x, x),
        index=0
    )
    
    metadata = predictor.metadata
    model_stats = metadata["models"].get(model_choice, {})
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    acc = model_stats.get("accuracy", 0.0)
    f1 = model_stats.get("weighted_f1", 0.0)
    st.metric(label="Validation Accuracy", value=f"{acc*100:.2f}%")
    st.metric(label="Weighted F1-Score", value=f"{f1:.3f}")
    
    st.markdown("---")
    st.markdown(f"**Total Categories:** {metadata['num_classes']}")
    st.markdown(f"**Training Set Size:** {metadata['train_size']} resumes")
    st.markdown(f"**Test Set Size:** {metadata['test_size']} resumes")

# Main Content
st.markdown('<div class="main-header">📄 AI Resume Screening & Category Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated candidate categorization across 24 industry domains with NLP feature extraction & ATS skill matching.</div>', unsafe_allow_html=True)

tabs = st.tabs(["🎯 Single Resume Screening", "📁 Bulk Candidate Batch", "📈 Model Analytics & EDA", "ℹ️ Architecture & Pipeline"])

# ==========================================
# TAB 1: Single Resume Screener
# ==========================================
with tabs[0]:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("1. Input Resume")
        
        input_mode = st.radio("Choose Input Method:", ["Upload File (PDF/DOCX/TXT)", "Paste Resume Text", "Quick Demo Sample"], horizontal=True)
        
        resume_text = ""
        
        if input_mode == "Upload File (PDF/DOCX/TXT)":
            uploaded_file = st.file_uploader("Upload candidate resume:", type=["pdf", "docx", "txt"])
            if uploaded_file is not None:
                try:
                    with st.spinner("Extracting text from file..."):
                        resume_text = extract_text_from_uploaded_file(uploaded_file)
                    st.success(f"Extracted {len(resume_text.split())} words from {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    
        elif input_mode == "Paste Resume Text":
            resume_text = st.text_area(
                "Paste candidate resume content here:",
                height=280,
                placeholder="Paste the resume text (experience, skills, summary, education)..."
            )
            
        else: # Quick Demo Sample
            sample_titles = [s["title"] for s in samples]
            selected_sample_title = st.selectbox("Select a preloaded sample:", sample_titles)
            selected_sample = next(s for s in samples if s["title"] == selected_sample_title)
            resume_text = st.text_area("Sample Content:", value=selected_sample["text"], height=240)
            
        analyze_btn = st.button("🚀 Classify & Analyze Resume", type="primary", use_container_width=True)
        
    with col2:
        st.subheader("2. Screening Intelligence")
        
        if analyze_btn and resume_text.strip():
            with st.spinner("Analyzing with NLP model and extracting skills..."):
                result = predictor.predict(resume_text, model_name=model_choice, top_k=5)
                
            pred_cat = result["predicted_category"]
            confidence = result["confidence"]
            conf_pct = result["confidence_percentage"]
            
            # Prediction Card
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 0.9rem; color: #6B7280; text-transform: uppercase; font-weight: 600;">Primary Match</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #1E40AF; margin-top: 4px;">{pred_cat}</div>
                <div style="font-size: 1.1rem; color: #047857; font-weight: 600; margin-top: 6px;">
                    Confidence Score: <b>{conf_pct}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Top-5 Predictions Probability Chart
            st.markdown("#### 📊 Category Probability Distribution")
            top_df = pd.DataFrame(result["top_predictions"])
            
            fig, ax = plt.subplots(figsize=(7, 3.2))
            colors = ["#2563EB" if i == 0 else "#93C5FD" for i in range(len(top_df))]
            bars = ax.barh(top_df["category"][::-1], top_df["probability"][::-1] * 100, color=colors[::-1])
            ax.set_xlabel("Probability (%)", fontsize=10)
            ax.set_xlim(0, 100)
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', ha='left', fontsize=9, fontweight='bold', color='#1E3A8A')
            sns.despine(top=True, right=True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Matched Skills / Keywords
            st.markdown("#### 🎯 Matched Role Skills (ATS Analysis)")
            matched_skills = result["matched_skills"]
            if matched_skills:
                badges_html = "".join([f'<span class="badge">✓ {s}</span>' for s in matched_skills])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.info("No standard category-specific keywords matched. Model classified based on general semantic context.")
                
            # Expandable Preprocessed Text
            with st.expander("🔍 View Preprocessed (Cleaned) Text fed to TF-IDF"):
                st.code(result["cleaned_text"])
                
        elif analyze_btn and not resume_text.strip():
            st.warning("⚠️ Please provide resume text or upload a document first.")
        else:
            st.info("👈 Upload a resume or select a sample on the left, then click 'Classify & Analyze Resume'.")


# ==========================================
# TAB 2: Batch Candidate Screener
# ==========================================
with tabs[1]:
    st.subheader("Bulk Resume Screening & Shortlisting")
    st.write("Upload multiple candidate resumes simultaneously to classify and rank them automatically.")
    
    bulk_files = st.file_uploader(
        "Upload multiple resume files (PDF, DOCX, TXT):",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if bulk_files:
        if st.button(f"⚡ Screen {len(bulk_files)} Candidates", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            resumes_data = []
            for i, f in enumerate(bulk_files):
                status_text.text(f"Processing candidate {i+1} of {len(bulk_files)}: {f.name}...")
                try:
                    text = extract_text_from_uploaded_file(f)
                    resumes_data.append({"name": f.name, "text": text})
                except Exception as e:
                    resumes_data.append({"name": f.name, "text": ""})
                progress_bar.progress((i + 1) / len(bulk_files))
                
            status_text.text("Generating predictions and ranking candidates...")
            batch_df = predictor.predict_batch(resumes_data, model_name=model_choice)
            status_text.empty()
            progress_bar.empty()
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Resumes Processed", len(batch_df))
            with m2:
                top_dom = batch_df["Predicted Category"].mode()[0] if not batch_df.empty else "-"
                st.metric("Top Domain Found", top_dom)
            with m3:
                st.metric("Model Used", model_choice)
                
            st.markdown("### 📋 Candidate Shortlist Table")
            st.dataframe(batch_df, use_container_width=True)
            
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Candidate Screening Report (CSV)",
                data=csv,
                file_name="candidate_screening_report.csv",
                mime="text/csv",
                type="secondary"
            )


# ==========================================
# TAB 3: Model Analytics & EDA
# ==========================================
with tabs[2]:
    st.subheader("Dataset Exploratory Data Analysis & Model Comparison")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📊 Dataset Category Distribution (2,484 Resumes)")
        if os.path.exists("models/category_distribution.png"):
            st.image("models/category_distribution.png", use_container_width=True)
        else:
            st.info("Distribution plot not found. Run training script to generate.")
            
    with col_b:
        st.markdown("#### 🏆 Model Benchmark Comparison")
        comp_data = []
        for m_name, m_info in metadata["models"].items():
            comp_data.append({
                "Model": m_name,
                "Test Accuracy": f"{m_info['accuracy']*100:.2f}%",
                "Macro F1-Score": f"{m_info['macro_f1']:.3f}",
                "Weighted F1-Score": f"{m_info['weighted_f1']:.3f}"
            })
        st.table(pd.DataFrame(comp_data))
        
        st.markdown("""
        **Key Takeaways:**
        - **Calibrated LinearSVC** outperforms standard Naive Bayes by **+14.3%** in test accuracy.
        - Linear SVM with sublinear TF-IDF scaling effectively separates sparse text boundaries in high-dimensional space (5,000 features across 24 classes).
        """)
        
    st.markdown("---")
    st.markdown("#### 🎯 Confusion Matrix Heatmap")
    cm_model = st.radio("Select Confusion Matrix to View:", ["CalibratedLinearSVC", "MultinomialNB"], horizontal=True)
    if cm_model == "CalibratedLinearSVC" and os.path.exists("models/confusion_matrix_svc.png"):
        st.image("models/confusion_matrix_svc.png", caption="LinearSVC Confusion Matrix", use_container_width=True)
    elif cm_model == "MultinomialNB" and os.path.exists("models/confusion_matrix_nb.png"):
        st.image("models/confusion_matrix_nb.png", caption="Multinomial Naive Bayes Confusion Matrix", use_container_width=True)


# ==========================================
# TAB 4: Architecture & Pipeline
# ==========================================
with tabs[3]:
    st.subheader("System Architecture & Processing Pipeline")
    
    st.markdown("""
    ```
    ┌─────────────────────────┐
    │ Raw Resume File / Text  │  (PDF, DOCX, TXT, Raw String)
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │  Document Text Parsing  │  (pypdf / python-docx stream extraction)
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │   Regex Preprocessing   │  (Remove URLs, mentions, hashtags, symbols, digits)
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │   TF-IDF Vectorization  │  (5,000 unigram/bigram features with sublinear TF)
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │   Model Classification  │  (Calibrated LinearSVC / Multinomial Naive Bayes)
    └───────────┬─────────────┘
                │
                ▼
    ┌────────────────────────────────────────────────────────┐
    │ Predicted Category (1 of 24) + Confidence + ATS Skills │
    └────────────────────────────────────────────────────────┘
    ```
    
    ### 📂 Supported Job Categories (24 Classes):
    """)
    
    cat_cols = st.columns(4)
    all_cats = sorted(list(CATEGORY_SKILLS.keys()))
    for idx, cat in enumerate(all_cats):
        with cat_cols[idx % 4]:
            st.markdown(f"- **{cat}**")
