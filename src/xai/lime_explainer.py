import numpy as np
import lime
import lime.lime_text
import logging
import sys
import os

# Add src to path for internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import prepare_full_features

logger = logging.getLogger(__name__)

class LimeExplainer:
    def __init__(self, model, model_type="classical", vectorizer=None):
        self.model = model
        self.model_type = model_type
        self.vectorizer = vectorizer
        self.lime_explainer = lime.lime_text.LimeTextExplainer(
            class_names=["legitimate", "phishing"], split_expression=r'\W+', bow=True)

    def get_explanation(self, text, num_features=15, num_samples=500):
        try:
            def predict_fn(texts):
                if self.model_type == "classical":
                    from core.utils import prepare_full_features
                    all_features = []
                    for t in texts:
                        feats, _ = prepare_full_features(t, self.vectorizer)
                        all_features.append(feats[0])
                    return self.model.predict_proba(np.array(all_features))
                return np.array([self._transformer_predict_proba(t) for t in texts])
            exp = self.lime_explainer.explain_instance(text, predict_fn,
                                                       num_features=num_features, num_samples=num_samples)
            return [{"token": tok, "weight": round(float(w), 6)} for tok, w in exp.as_list()]
        except Exception as e:
            logger.error(f"LIME failed: {e}")
            return [{"token": "unavailable", "weight": 0.0}]

    def _transformer_predict_proba(self, text):
        try:
            if hasattr(self.model, '__call__'):
                result = self.model(text)
                if isinstance(result, list):
                    scores = {r['label']: r['score'] for r in result}
                    return np.array([scores.get('LABEL_0', scores.get('legitimate', 0.5)),
                                     scores.get('LABEL_1', scores.get('phishing', 0.5))])
        except Exception:
            pass
        return np.array([0.5, 0.5])

    def visualize(self, text):
        """Placeholder for visualization function"""
        pass
