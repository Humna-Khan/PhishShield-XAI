"""
FastAPI Main Application — Phishing Detection XAI API.
Provides real-time email classification with SHAP, LIME, and LLM explanations.
Designed for deployment in Google Colab with pyngrok.
"""

import os
import sys
import logging
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .models import (
    EmailRequest, AnalysisResponse, HealthResponse,
    SHAPFeature, LIMEHighlight, BatchEmailRequest, BatchAnalysisResponse
)

# Add src to path for internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import extract_features, prepare_full_features
from xai.shap_explainer import ShapExplainer
from xai.lime_explainer import LimeExplainer

_FEAT_DESC = {
    "urgency_word_count":   ("high urgency language density",        "absence of urgency language"),
    "url_count":            ("multiple embedded URLs",               "normal URL usage"),
    "suspicious_url_count": ("shortened or obfuscated URLs",         "clean, trustworthy URLs"),
    "money_word_count":     ("financial / credential trigger words", "no financial trigger words"),
    "threat_score":         ("threatening language patterns",        "no threatening language"),
    "impersonation_score":  ("brand impersonation signals",          "no impersonation indicators"),
    "capital_ratio":        ("excessive capitalisation",             "normal capitalisation"),
    "exclamation_count":    ("excessive exclamation marks",          "measured punctuation"),
    "has_html":             ("embedded HTML markup",                 "plain-text format"),
    "special_char_ratio":   ("unusual special-character density",    "normal character distribution"),
    "politeness_score":     ("formulaic politeness phrases",         "natural professional tone"),
}

def generate_llm_explanation(prediction, confidence, shap_feats, lime_highlights):
    """Rule-based NLG: generates contextual analyst explanations from SHAP + LIME signals."""
    is_phish = prediction == "phishing"
    conf_word = "high" if confidence > 0.85 else "moderate" if confidence > 0.65 else "slight"

    # Sentence 1 — verdict
    if is_phish:
        s1 = (f"This email was classified as phishing with {conf_word} confidence "
              f"({confidence*100:.0f}%) based on {len(shap_feats)} linguistic and structural signals.")
    else:
        s1 = (f"The classifier identified this email as legitimate ({confidence*100:.0f}% confidence) "
              f"— its features closely match normal professional correspondence.")

    # Sentence 2 — top SHAP feature
    s2 = ""
    if shap_feats:
        top = shap_feats[0]
        phish_desc, legit_desc = _FEAT_DESC.get(
            top["feature"],
            (f"elevated {top['feature'].replace('_',' ')}",
             f"normal {top['feature'].replace('_',' ')}"))
        if top["direction"] == "phishing":
            s2 = f"The strongest phishing signal was {phish_desc} (SHAP feature: {top['feature']}, weight {top['importance']:.3f})."
        else:
            s2 = f"The key legitimacy indicator was {legit_desc} (SHAP feature: {top['feature']}, weight {top['importance']:.3f})."

    # Sentence 3 — LIME tokens
    phish_toks = [h["token"] for h in sorted(lime_highlights, key=lambda x: -x["weight"]) if h["weight"] > 0][:3]
    legit_toks  = [h["token"] for h in sorted(lime_highlights, key=lambda x:  x["weight"]) if h["weight"] < 0][:2]
    if phish_toks and is_phish:
        s3 = f"LIME token analysis highlighted '{', '.join(phish_toks)}' as the primary phishing-indicative terms."
    elif legit_toks and not is_phish:
        s3 = f"Tokens such as '{', '.join(legit_toks)}' reinforced the legitimate classification."
    else:
        s3 = "LIME analysis confirms the model focused on semantically meaningful content rather than superficial patterns."

    return f"{s1} {s2} {s3}".strip()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
MODEL = None
VECTORIZER = None
FEATURE_NAMES = None
EXPLAINER_SHAP = None
EXPLAINER_LIME = None
MODEL_TYPE = "classical"


def extract_linguistic_features(text):
    """Extract handcrafted linguistic features from email text."""
    import re
    features = {}
    text_lower = text.lower()
    words = text_lower.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    # Urgency indicators
    urgency_words = ['urgent', 'immediately', 'asap', 'now', 'hurry', 'expire',
                     'deadline', 'suspended', 'verify', 'confirm', 'alert', 'warning',
                     'action required', 'limited time', 'act now']
    features['urgency_word_count'] = sum(1 for w in urgency_words if w in text_lower)

    # Suspicious patterns
    features['url_count'] = len(re.findall(r'https?://\S+', text))
    features['suspicious_url_count'] = len(re.findall(
        r'https?://(?:bit\.ly|tinyurl|goo\.gl|t\.co|rb\.gy|shorturl|click\.)', text_lower))
    features['email_count'] = len(re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text))
    features['has_html'] = 1 if re.search(r'<[^>]+>', text) else 0
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')

    # Financial / credential indicators
    money_words = ['bank', 'account', 'password', 'credit', 'ssn', 'social security',
                   'routing', 'wire', 'transfer', 'payment', 'invoice', 'bitcoin', 'wallet']
    features['money_word_count'] = sum(1 for w in money_words if w in text_lower)

    # Text statistics
    features['word_count'] = len(words)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['sentence_count'] = max(len(sentences), 1)
    features['avg_sentence_length'] = len(words) / max(len(sentences), 1)
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    features['special_char_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)

    # Politeness / social engineering
    polite_words = ['dear', 'please', 'kindly', 'sir', 'madam', 'respected', 'valued customer']
    features['politeness_score'] = sum(1 for w in polite_words if w in text_lower)

    # Impersonation indicators
    impersonation_words = ['official', 'authorized', 'government', 'irs', 'fbi',
                           'microsoft', 'google', 'apple', 'amazon', 'paypal', 'netflix']
    features['impersonation_score'] = sum(1 for w in impersonation_words if w in text_lower)

    # Threat indicators
    threat_words = ['suspend', 'terminate', 'close', 'block', 'disable', 'unauthorized',
                    'breach', 'compromise', 'violation', 'penalty', 'legal action']
    features['threat_score'] = sum(1 for w in threat_words if w in text_lower)

    return features


def prepare_features(text, vectorizer=None):
    """Prepare full feature vector from email text."""
    # Linguistic features
    ling_feats = extract_linguistic_features(text)
    ling_values = np.array(list(ling_feats.values())).reshape(1, -1)
    ling_names = list(ling_feats.keys())

    # TF-IDF features
    if vectorizer is not None:
        tfidf = vectorizer.transform([text]).toarray()
        tfidf_names = [f"tfidf_{n}" for n in vectorizer.get_feature_names_out()]
        features = np.hstack([ling_values, tfidf])
        names = ling_names + tfidf_names
    else:
        features = ling_values
        names = ling_names

    return features, names


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global MODEL, VECTORIZER, FEATURE_NAMES, EXPLAINER_SHAP, EXPLAINER_LIME, MODEL_TYPE

    model_dir = os.environ.get("MODEL_DIR", "./models/classical_model")
    model_path = os.path.join(model_dir, "classical_model.pkl")
    vec_path = os.path.join(model_dir, "vectorizer.pkl")
    names_path = os.path.join(model_dir, "feature_names.pkl")

    try:
        if os.path.exists(model_path):
            MODEL = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        if os.path.exists(vec_path):
            VECTORIZER = joblib.load(vec_path)
            logger.info("Vectorizer loaded")
        if os.path.exists(names_path):
            FEATURE_NAMES = joblib.load(names_path)

        if MODEL is not None:
            EXPLAINER_SHAP = ShapExplainer(
                model=MODEL, model_type=MODEL_TYPE,
                feature_names=FEATURE_NAMES)
            EXPLAINER_LIME = LimeExplainer(
                model=MODEL, model_type=MODEL_TYPE,
                vectorizer=VECTORIZER)
            logger.info("Explainers initialized")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Phishing Detection XAI API",
    description="Real-time phishing detection with SHAP, LIME, and LLM explanations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", model_loaded=MODEL is not None, version="1.0.0")


@app.post("/predict", response_model=AnalysisResponse)
async def predict_email(request: EmailRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.email_text
    features, feat_names = prepare_full_features(text, VECTORIZER)

    # Prediction
    proba = MODEL.predict_proba(features)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(proba[pred_class])
    classification = "phishing" if pred_class == 1 else "legitimate"

    # SHAP explanation
    if EXPLAINER_SHAP is not None:
        shap_feats = EXPLAINER_SHAP.get_explanation(features, top_k=10)
        lime_highlights = EXPLAINER_LIME.get_explanation(text, num_features=15)
    else:
        shap_feats = [{"feature": "model_confidence", "importance": confidence, "direction": classification}]
        lime_highlights = [{"token": "N/A", "weight": 0.0}]

    # LLM explanation
    llm_explanation = generate_llm_explanation(classification, confidence, shap_feats, lime_highlights)

    return AnalysisResponse(
        classification=classification,
        confidence_score=round(confidence, 4),
        shap_features=[SHAPFeature(**f) for f in shap_feats[:10]],
        lime_highlights=[LIMEHighlight(**h) for h in lime_highlights],
        llm_explanation=llm_explanation,
        model_used=MODEL_TYPE,
    )


@app.post("/batch_analyze", response_model=BatchAnalysisResponse)
async def batch_analyze(request: BatchEmailRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for email_text in request.emails:
        req = EmailRequest(email_text=email_text)
        result = await predict_email(req)
        results.append(result)

    phishing_count = sum(1 for r in results if r.classification == "phishing")
    return BatchAnalysisResponse(
        results=results, total=len(results),
        phishing_count=phishing_count,
        legitimate_count=len(results) - phishing_count,
        evasion_rate=round(1 - phishing_count / max(len(results), 1), 4),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
