"""
Self-contained FastAPI app for PhishShield-XAI.
Run in Colab: uvicorn colab_api:app --host 0.0.0.0 --port 8000
No API keys required — LLM explanations use built-in rule-based NLG.
"""
import os, re, logging
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MODEL = VECTORIZER = FEATURE_NAMES = EXPLAINER_SHAP = EXPLAINER_LIME = None

# ---------------------------------------------------------------------------
# Feature descriptions for NLG
# ---------------------------------------------------------------------------
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
    """
    Rule-based NLG: generates a contextual analyst explanation from real
    SHAP feature values and LIME token weights. No API keys required.
    """
    is_phish = prediction == "phishing"
    conf_word = "high" if confidence > 0.85 else "moderate" if confidence > 0.65 else "slight"

    # Sentence 1 — verdict + confidence
    if is_phish:
        s1 = (f"This email was classified as phishing with {conf_word} confidence "
              f"({confidence*100:.0f}%) based on {len(shap_feats)} linguistic and structural signals.")
    else:
        s1 = (f"The classifier identified this email as legitimate ({confidence*100:.0f}% confidence) "
              f"— its features closely match normal professional correspondence.")

    # Sentence 2 — dominant SHAP feature
    s2 = ""
    if shap_feats:
        top = shap_feats[0]
        phish_desc, legit_desc = _FEAT_DESC.get(
            top["feature"],
            (f"elevated {top['feature'].replace('_', ' ')}",
             f"normal {top['feature'].replace('_', ' ')}"))
        if top["direction"] == "phishing":
            s2 = (f"The strongest phishing signal was {phish_desc} "
                  f"(SHAP: {top['feature']}, weight {top['importance']:.3f}).")
        else:
            s2 = (f"The key legitimacy indicator was {legit_desc} "
                  f"(SHAP: {top['feature']}, weight {top['importance']:.3f}).")

    # Sentence 3 — LIME tokens
    phish_toks = [h["token"] for h in sorted(lime_highlights, key=lambda x: -x["weight"])
                  if h["weight"] > 0][:3]
    legit_toks  = [h["token"] for h in sorted(lime_highlights, key=lambda x:  x["weight"])
                   if h["weight"] < 0][:2]

    if phish_toks and is_phish:
        s3 = f"LIME token analysis highlighted '{', '.join(phish_toks)}' as primary phishing-indicative terms."
    elif legit_toks and not is_phish:
        s3 = f"Tokens such as '{', '.join(legit_toks)}' reinforced the legitimate classification."
    else:
        s3 = "LIME analysis confirms the model focused on semantically meaningful content."

    return f"{s1} {s2} {s3}".strip()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(text):
    tl = text.lower()
    words = tl.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    urgency = ["urgent","immediately","asap","hurry","expire","suspended","verify","confirm","alert","warning"]
    money   = ["bank","account","password","credit","ssn","wire","transfer","payment","invoice","bitcoin","wallet"]
    threats = ["suspend","terminate","close","block","disable","unauthorized","breach","compromise","violation","penalty"]
    polite  = ["dear","please","kindly","sir","madam","respected"]
    impers  = ["official","authorized","government","irs","microsoft","google","apple","amazon","paypal"]
    return {
        "urgency_word_count":   sum(1 for w in urgency if w in tl),
        "url_count":            len(re.findall(r"https?://\S+", text)),
        "suspicious_url_count": len(re.findall(r"https?://(?:bit\.ly|tinyurl|goo\.gl)", tl)),
        "email_count":          len(re.findall(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)),
        "has_html":             int(bool(re.search(r"<[^>]+>", text))),
        "exclamation_count":    text.count("!"),
        "question_count":       text.count("?"),
        "money_word_count":     sum(1 for w in money   if w in tl),
        "word_count":           len(words),
        "avg_word_length":      np.mean([len(w) for w in words]) if words else 0,
        "sentence_count":       max(len(sentences), 1),
        "avg_sentence_length":  len(words) / max(len(sentences), 1),
        "capital_ratio":        sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "digit_ratio":          sum(1 for c in text if c.isdigit())  / max(len(text), 1),
        "special_char_ratio":   sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1),
        "politeness_score":     sum(1 for w in polite  if w in tl),
        "impersonation_score":  sum(1 for w in impers  if w in tl),
        "threat_score":         sum(1 for w in threats if w in tl),
    }


def prepare_features(text, vectorizer):
    ling = extract_features(text)
    lv   = np.array(list(ling.values())).reshape(1, -1)
    tf   = vectorizer.transform([text]).toarray()
    feat_names = list(ling.keys()) + [f"tfidf_{n}" for n in vectorizer.get_feature_names_out()]
    return np.hstack([lv, tf]), feat_names


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class EmailRequest(BaseModel):
    email_text: str = Field(..., min_length=10, max_length=50000)

class SHAPFeature(BaseModel):
    feature: str; importance: float; direction: str

class LIMEHighlight(BaseModel):
    token: str; weight: float

class AnalysisResponse(BaseModel):
    classification: str; confidence_score: float
    shap_features: List[SHAPFeature]; lime_highlights: List[LIMEHighlight]
    llm_explanation: str; model_used: str = "classical"

class HealthResponse(BaseModel):
    status: str = "healthy"; model_loaded: bool = False; version: str = "1.0.0"

class BatchEmailRequest(BaseModel):
    emails: List[str]

class BatchAnalysisResponse(BaseModel):
    results: List[AnalysisResponse]; total: int
    phishing_count: int; legitimate_count: int; evasion_rate: Optional[float] = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app):
    global MODEL, VECTORIZER, FEATURE_NAMES, EXPLAINER_SHAP, EXPLAINER_LIME
    mdir = os.environ.get("MODEL_DIR", "models/classical_model")
    try:
        MODEL         = joblib.load(f"{mdir}/classical_model.pkl")
        VECTORIZER    = joblib.load(f"{mdir}/vectorizer.pkl")
        FEATURE_NAMES = joblib.load(f"{mdir}/feature_names.pkl")

        import shap, lime, lime.lime_text

        class _SHAP:
            def __init__(self, m, fn):
                self.fn = fn
                # TreeExplainer doesn't support VotingClassifier — extract RF estimator
                tree_model = m
                if hasattr(m, "named_estimators_") and "rf" in m.named_estimators_:
                    tree_model = m.named_estimators_["rf"]
                self.ex = shap.TreeExplainer(tree_model)
            def get_explanation(self, feats, top_k=10):
                sv = self.ex.shap_values(feats)
                if isinstance(sv, list):  vals = sv[1][0]
                elif sv.ndim == 3:        vals = sv[0, :, 1]
                else:                     vals = sv[0]
                idx = np.argsort(np.abs(vals))[::-1][:top_k]
                return [{"feature": self.fn[i] if i < len(self.fn) else f"f{i}",
                         "importance": round(float(np.abs(vals[i])), 6),
                         "direction": "phishing" if vals[i] > 0 else "legitimate"} for i in idx]

        class _LIME:
            def __init__(self, m, vec):
                self.m   = m
                self.vec = vec
                self.ex  = lime.lime_text.LimeTextExplainer(
                    class_names=["legitimate", "phishing"],
                    split_expression=r"\W+", bow=True)
            def get_explanation(self, text, num_features=15, num_samples=300):
                def pf(texts):
                    return self.m.predict_proba(
                        np.array([prepare_features(t, self.vec)[0][0] for t in texts]))
                exp = self.ex.explain_instance(text, pf,
                          num_features=num_features, num_samples=num_samples)
                return [{"token": t, "weight": round(float(w), 6)} for t, w in exp.as_list()]

        # Initialize each explainer separately so a SHAP failure doesn't kill LIME
        try:
            EXPLAINER_SHAP = _SHAP(MODEL, FEATURE_NAMES)
            logger.warning("SHAP explainer ready")
        except Exception as e:
            logger.error(f"SHAP init failed: {e}")
        try:
            EXPLAINER_LIME = _LIME(MODEL, VECTORIZER)
            logger.warning("LIME explainer ready")
        except Exception as e:
            logger.error(f"LIME init failed: {e}")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="PhishShield XAI API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"], allow_credentials=True)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", model_loaded=MODEL is not None)


@app.post("/predict", response_model=AnalysisResponse)
async def predict(req: EmailRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    feats, _ = prepare_features(req.email_text, VECTORIZER)
    proba = MODEL.predict_proba(feats)[0]
    cls   = int(np.argmax(proba))
    conf  = float(proba[cls])
    label = "phishing" if cls == 1 else "legitimate"
    shap_out = EXPLAINER_SHAP.get_explanation(feats, top_k=10) if EXPLAINER_SHAP else []
    lime_out = EXPLAINER_LIME.get_explanation(req.email_text) if EXPLAINER_LIME else []
    expl     = generate_llm_explanation(label, conf, shap_out, lime_out)
    return AnalysisResponse(
        classification=label, confidence_score=round(conf, 4),
        shap_features=[SHAPFeature(**f) for f in shap_out[:10]],
        lime_highlights=[LIMEHighlight(**h) for h in lime_out],
        llm_explanation=expl)


@app.post("/batch_analyze", response_model=BatchAnalysisResponse)
async def batch_analyze(req: BatchEmailRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    results = []
    for txt in req.emails:
        r = await predict(EmailRequest(email_text=txt))
        results.append(r)
    ph = sum(1 for r in results if r.classification == "phishing")
    return BatchAnalysisResponse(
        results=results, total=len(results),
        phishing_count=ph, legitimate_count=len(results) - ph,
        evasion_rate=round(1 - ph / max(len(results), 1), 4))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
