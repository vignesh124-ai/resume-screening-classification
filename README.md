# 📄 AI Resume Screening & Category Classification

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to automate resume screening and categorize candidate profiles across **24 industry domains**. Includes a modern, interactive **Streamlit Web Application** supporting PDF/DOCX/TXT file uploads, ATS keyword extraction, and batch candidate screening.

---

## 🌟 Key Features

- **24 Industry Job Domains:** Covers Information Technology, Engineering, HR, Finance, Healthcare, Sales, Teacher, Designer, Advocate, Chef, Aviation, and more.
- **Dual Classification Engines:**
  - **Enhanced Model (Calibrated LinearSVC):** Achieves **~70% test accuracy** with calibrated probability outputs.
  - **Baseline Model (Multinomial Naive Bayes):** Reproduces the original textbook pipeline (**~56% test accuracy**).
  - **Logistic Regression Model:** Solid multi-class linear benchmark (**~65% test accuracy**).
- **Automated ATS Skill Extraction:** Matches candidate qualifications against domain-specific skill dictionaries.
- **Multi-Format Document Parsing:** Ingests raw text or extracts from `.pdf`, `.docx`, and `.txt` files directly.
- **Bulk Candidate Screening:** Upload multiple candidate resumes at once, automatically rank them by category and confidence, and export the shortlist as a CSV.
- **Interactive Web UI:** Built with Streamlit, featuring real-time probability charts, confidence badges, and EDA visualizations.

---

## 📊 Model Performance Comparison

Evaluated on an 80/20 train/test split across 2,484 candidate resumes:

| Model Architecture | Accuracy | Macro F1-Score | Weighted F1-Score | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Calibrated LinearSVC** | **69.82%** | **0.658** | **0.690** | **Best overall (Recommended)** |
| **Logistic Regression** | **65.19%** | **0.612** | **0.645** | Strong linear baseline |
| **Multinomial Naive Bayes** | **55.53%** | **0.485** | **0.542** | Original baseline model |

---

## 🏗️ System Architecture & Pipeline

```
┌─────────────────────────────────┐
│     Resume Input (PDF/DOCX/TXT) │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│      Document Text Extraction   │  (pypdf / python-docx / UTF-8)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       Regex Cleaning Pipeline   │  (Strip URLs, tags, mentions, symbols, numbers)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       TF-IDF Vectorization      │  (5,000 features, sublinear TF scaling)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Classification & Probabilities │ (Calibrated LinearSVC / Naive Bayes)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Domain Category + ATS Skills   │  (Ranked top matches & candidate analytics)
└─────────────────────────────────┘
```

---

## 📁 Repository Structure

```
resume-screening-classification/
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation and usage guide
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── Resume.csv              # Full dataset (2,484 resumes, 24 classes)
│   └── sample_resumes.json     # Curated samples for one-click testing
│
├── models/
│   ├── tfidf_vectorizer.joblib # Trained TF-IDF feature extractor
│   ├── label_encoder.joblib    # Fitted label encoder
│   ├── model_svc.joblib        # Calibrated LinearSVC (Enhanced model)
│   ├── model_nb.joblib         # Multinomial Naive Bayes (Baseline)
│   ├── model_lr.joblib         # Logistic Regression model
│   ├── model_metadata.json     # Benchmark metrics and class listings
│   ├── category_distribution.png
│   ├── confusion_matrix_svc.png
│   └── confusion_matrix_nb.png
│
├── notebooks/
│   └── Resume_Classification.ipynb # Executable Jupyter notebook with EDA & modeling
│
└── src/
    ├── __init__.py
    ├── preprocessor.py         # Text cleaning and PDF/DOCX extractors
    ├── skills_dictionary.py    # Category-specific skill keywords for ATS matching
    ├── train.py                # End-to-end training and evaluation script
    └── predictor.py            # High-level inference engine
```

---

## 🚀 Getting Started

### 1. Installation

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application

Run the Streamlit application:
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 3. Retrain the Models (Optional)

To retrain and regenerate all model weights, evaluation plots, and metadata:
```bash
python -m src.train
```

### 4. Run the Jupyter Notebook

Open and interact with the full exploratory data analysis and modeling notebook:
```bash
jupyter notebook notebooks/Resume_Classification.ipynb
```

---

## 💼 Supported Job Categories (24)

- `ACCOUNTANT`
- `ADVOCATE`
- `AGRICULTURE`
- `APPAREL`
- `ARTS`
- `AUTOMOBILE`
- `AVIATION`
- `BANKING`
- `BPO`
- `BUSINESS-DEVELOPMENT`
- `CHEF`
- `CONSTRUCTION`
- `CONSULTANT`
- `DESIGNER`
- `DIGITAL-MEDIA`
- `ENGINEERING`
- `FINANCE`
- `FITNESS`
- `HEALTHCARE`
- `HR`
- `INFORMATION-TECHNOLOGY`
- `PUBLIC-RELATIONS`
- `SALES`
- `TEACHER`
