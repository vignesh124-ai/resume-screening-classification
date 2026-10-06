import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from src.preprocessor import clean_resume
from src.skills_dictionary import extract_matched_skills

class ResumePredictor:
    """Production-ready inference engine for Resume Classification."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        
        # Load artifacts
        self.tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
        self.le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
        
        self.models = {
            "CalibratedLinearSVC": joblib.load(os.path.join(models_dir, "model_svc.joblib")),
            "MultinomialNB": joblib.load(os.path.join(models_dir, "model_nb.joblib")),
            "LogisticRegression": joblib.load(os.path.join(models_dir, "model_lr.joblib"))
        }
        
        with open(os.path.join(models_dir, "model_metadata.json"), "r") as f:
            self.metadata = json.load(f)
            
        self.classes = list(self.le.classes_)
        
    def predict(
        self,
        text: str,
        model_name: str = "CalibratedLinearSVC",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Predict category, confidence scores, and extract matched skills for a single resume.
        """
        if model_name not in self.models:
            model_name = "CalibratedLinearSVC"
            
        model = self.models[model_name]
        cleaned = clean_resume(text)
        
        if not cleaned:
            return {
                "predicted_category": "Unknown",
                "confidence": 0.0,
                "top_predictions": [],
                "matched_skills": {},
                "cleaned_text": ""
            }
            
        vector = self.tfidf.transform([cleaned])
        
        # Get probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vector)[0]
        else:
            # Fallback for models without predict_proba
            decision = model.decision_function(vector)[0]
            exp_dec = np.exp(decision - np.max(decision))
            probs = exp_dec / exp_dec.sum()
            
        best_idx = int(np.argmax(probs))
        predicted_cat = self.classes[best_idx]
        confidence = float(probs[best_idx])
        
        # Top-K predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_predictions = [
            {
                "category": self.classes[idx],
                "probability": float(probs[idx]),
                "percentage": f"{probs[idx] * 100:.2f}%"
            }
            for idx in top_indices
        ]
        
        # Matched skills
        skills = extract_matched_skills(text, predicted_cat)
        matched_list = skills.get(predicted_cat, [])
        
        return {
            "predicted_category": predicted_cat,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "top_predictions": top_predictions,
            "matched_skills": matched_list,
            "all_skills_detected": extract_matched_skills(text),
            "cleaned_text": cleaned
        }

    def predict_batch(
        self,
        resumes: List[Dict[str, str]],
        model_name: str = "CalibratedLinearSVC"
    ) -> pd.DataFrame:
        """
        Batch prediction for multiple candidates.
        resumes: List of dicts with 'name' (or 'id') and 'text'.
        """
        results = []
        for r in resumes:
            name = r.get("name", "Unknown")
            raw_text = r.get("text", "")
            pred = self.predict(raw_text, model_name=model_name, top_k=3)
            
            runner_up = pred["top_predictions"][1]["category"] if len(pred["top_predictions"]) > 1 else "-"
            runner_up_conf = pred["top_predictions"][1]["percentage"] if len(pred["top_predictions"]) > 1 else "-"
            
            results.append({
                "Candidate / File": name,
                "Predicted Category": pred["predicted_category"],
                "Confidence": pred["confidence_percentage"],
                "Alternative Match": f"{runner_up} ({runner_up_conf})",
                "Key Skills Found": ", ".join(pred["matched_skills"][:5]) if pred["matched_skills"] else "None detected",
                "Word Count": len(raw_text.split())
            })
            
        return pd.DataFrame(results)
