import numpy as np
import shap
import logging

logger = logging.getLogger(__name__)

class ShapExplainer:
    def __init__(self, model, model_type="classical", background_data=None, feature_names=None):
        self.model = model
        self.model_type = model_type
        self.background_data = background_data
        self.feature_names = feature_names
        self.shap_explainer = self._init_shap_explainer()

    def _init_shap_explainer(self):
        try:
            if self.model_type == "classical":
                return shap.TreeExplainer(self.model)
            elif self.background_data is not None:
                pred_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model
                return shap.KernelExplainer(pred_fn, self.background_data[:100])
            else:
                return None
        except Exception as e:
            logger.error(f"SHAP init failed: {e}")
            return None

    def get_explanation(self, features, top_k=10):
        try:
            if self.shap_explainer is None:
                return self._fallback_feature_importance(features, top_k)
            shap_values = self.shap_explainer.shap_values(features)
            if isinstance(shap_values, list):
                values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif isinstance(shap_values, np.ndarray):
                values = shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[0]
            else:
                values = np.array(shap_values.values[0])
            names = self.feature_names or [f"feature_{i}" for i in range(len(values))]
            indices = np.argsort(np.abs(values))[::-1][:top_k]
            return [{"feature": names[idx] if idx < len(names) else f"feature_{idx}",
                     "importance": round(float(np.abs(values[idx])), 6),
                     "direction": "phishing" if values[idx] > 0 else "legitimate"} for idx in indices]
        except Exception as e:
            logger.error(f"SHAP failed: {e}")
            return self._fallback_feature_importance(features, top_k)

    def _fallback_feature_importance(self, features, top_k):
        try:
            if hasattr(self.model, 'feature_importances_'):
                imp = self.model.feature_importances_
                names = self.feature_names or [f"feature_{i}" for i in range(len(imp))]
                indices = np.argsort(imp)[::-1][:top_k]
                return [{"feature": names[i], "importance": round(float(imp[i]), 6),
                         "direction": "phishing" if features[0][i] > 0 else "legitimate"} for i in indices]
        except Exception:
            pass
        return [{"feature": "unavailable", "importance": 0.0, "direction": "unknown"}]

    def visualize(self, features):
        """Placeholder for visualization function"""
        pass
