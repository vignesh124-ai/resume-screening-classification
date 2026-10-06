import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocessor import clean_resume

def run_training(
    data_path: str = "data/Resume.csv",
    output_dir: str = "models",
    random_state: int = 42
):
    print("=" * 60)
    print(" RESUME CLASSIFICATION TRAINING PIPELINE")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"[1/6] Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"      Total records loaded: {len(df)}")
    print(f"      Columns: {list(df.columns)}")
    print(f"      Unique categories: {df['Category'].nunique()}")
    
    # Generate Category Distribution Plot
    plt.figure(figsize=(12, 8))
    sns.countplot(y=df['Category'], order=df['Category'].value_counts().index, palette="viridis")
    plt.title("Resume Category Distribution (24 Categories)", fontsize=14, fontweight="bold")
    plt.xlabel("Sample Count", fontsize=12)
    plt.ylabel("Job Category", fontsize=12)
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "category_distribution.png")
    plt.savefig(dist_plot_path, dpi=200)
    plt.close()
    print(f"      Saved category distribution plot -> {dist_plot_path}")
    
    # 2. Text Preprocessing
    print("[2/6] Cleaning resume text...")
    df['Cleaned_Resume'] = df['Resume_str'].apply(clean_resume)
    
    # 3. Label Encoding & Train-Test Split
    print("[3/6] Encoding target categories and splitting data (80/20)...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['Category'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['Cleaned_Resume'],
        y_encoded,
        test_size=0.2,
        random_state=random_state,
        stratify=y_encoded
    )
    print(f"      Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 4. TF-IDF Feature Extraction
    print("[4/6] Extracting TF-IDF features (max_features=5000, sublinear_tf=True)...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"      TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    
    # 5. Model Training & Evaluation
    print("[5/6] Training models...")
    
    # A) Multinomial Naive Bayes (Baseline from original project)
    print("      -> Training Multinomial Naive Bayes (Baseline)...")
    model_nb = MultinomialNB(alpha=0.1)
    model_nb.fit(X_train_tfidf, y_train)
    y_pred_nb = model_nb.predict(X_test_tfidf)
    acc_nb = float(accuracy_score(y_test, y_pred_nb))
    rep_nb = classification_report(y_test, y_pred_nb, target_names=le.classes_, output_dict=True, zero_division=0)
    print(f"         MultinomialNB Test Accuracy: {acc_nb:.4f} ({acc_nb*100:.2f}%)")
    
    # B) Calibrated Linear Support Vector Machine (High Accuracy)
    print("      -> Training Calibrated LinearSVC (Enhanced Model)...")
    base_svc = LinearSVC(C=1.0, dual='auto', random_state=random_state)
    model_svc = CalibratedClassifierCV(base_svc, cv=3)
    model_svc.fit(X_train_tfidf, y_train)
    y_pred_svc = model_svc.predict(X_test_tfidf)
    acc_svc = float(accuracy_score(y_test, y_pred_svc))
    rep_svc = classification_report(y_test, y_pred_svc, target_names=le.classes_, output_dict=True, zero_division=0)
    print(f"         Calibrated LinearSVC Test Accuracy: {acc_svc:.4f} ({acc_svc*100:.2f}%)")
    
    # C) Logistic Regression
    print("      -> Training Logistic Regression...")
    model_lr = LogisticRegression(max_iter=1000, C=2.0, random_state=random_state)
    model_lr.fit(X_train_tfidf, y_train)
    y_pred_lr = model_lr.predict(X_test_tfidf)
    acc_lr = float(accuracy_score(y_test, y_pred_lr))
    rep_lr = classification_report(y_test, y_pred_lr, target_names=le.classes_, output_dict=True, zero_division=0)
    print(f"         Logistic Regression Test Accuracy: {acc_lr:.4f} ({acc_lr*100:.2f}%)")
    
    # Confusion Matrix for Calibrated LinearSVC
    cm_svc = confusion_matrix(y_test, y_pred_svc)
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm_svc,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.title(f"Confusion Matrix - Calibrated LinearSVC (Accuracy: {acc_svc*100:.1f}%)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix_svc.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    
    # Confusion Matrix for Naive Bayes
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm_nb,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.title(f"Confusion Matrix - Multinomial Naive Bayes (Accuracy: {acc_nb*100:.1f}%)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    cm_nb_path = os.path.join(output_dir, "confusion_matrix_nb.png")
    plt.savefig(cm_nb_path, dpi=200)
    plt.close()

    # 6. Save Artifacts
    print("[6/6] Saving model artifacts...")
    joblib.dump(tfidf, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(le, os.path.join(output_dir, "label_encoder.joblib"))
    joblib.dump(model_nb, os.path.join(output_dir, "model_nb.joblib"))
    joblib.dump(model_svc, os.path.join(output_dir, "model_svc.joblib"))
    joblib.dump(model_lr, os.path.join(output_dir, "model_lr.joblib"))
    
    metadata = {
        "classes": list(le.classes_),
        "num_classes": len(le.classes_),
        "dataset_size": len(df),
        "test_size": len(X_test),
        "train_size": len(X_train),
        "models": {
            "MultinomialNB": {
                "accuracy": acc_nb,
                "macro_f1": rep_nb["macro avg"]["f1-score"],
                "weighted_f1": rep_nb["weighted avg"]["f1-score"],
                "filename": "model_nb.joblib"
            },
            "CalibratedLinearSVC": {
                "accuracy": acc_svc,
                "macro_f1": rep_svc["macro avg"]["f1-score"],
                "weighted_f1": rep_svc["weighted avg"]["f1-score"],
                "filename": "model_svc.joblib"
            },
            "LogisticRegression": {
                "accuracy": acc_lr,
                "macro_f1": rep_lr["macro avg"]["f1-score"],
                "weighted_f1": rep_lr["weighted avg"]["f1-score"],
                "filename": "model_lr.joblib"
            }
        }
    }
    
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("=" * 60)
    print(" TRAINING COMPLETE!")
    print(f" Artifacts saved to: {output_dir}")
    print(f" Best Model: Calibrated LinearSVC with {acc_svc*100:.2f}% accuracy")
    print("=" * 60)
    return metadata

if __name__ == "__main__":
    run_training()
