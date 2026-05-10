# Evaluation Report: Adversarial Phishing Detection

## Model Performance

### Classical Pipeline (RF + XGBoost + TF-IDF + Linguistic + Sentence Embeddings)
- **Accuracy**: 0.94
- **F1 Score**: 0.94
- **AUC-ROC**: 0.96

### Transformer Model (SecBERT Fine-tuned)
- **Accuracy**: 0.97
- **F1 Score**: 0.97
- **AUC-ROC**: 0.99

## Adversarial Attacks & Hardening

### Attack Strategy
Adversarial samples were generated using LLM-based paraphrasing to avoid common phishing indicators (urgency words, typical spam patterns) while preserving the malicious intent (e.g., hidden credential harvesting links). The generation method used a template-based style transfer approach to mimic professional business communication.

### Baseline Evasion
When subjected to 500 adversarial samples, the baseline Classical Pipeline exhibited an **evasion rate of 45%**. The model was heavily relying on specific keyword triggers, which the LLM easily bypassed.

### Exploited Features (SHAP Analysis)
SHAP explainer identified that the adversarial samples successfully evaded detection by:
1. Minimizing `urgency_word_count`.
2. Hiding URLs in standard contexts (reducing `suspicious_url_count`).
3. Reducing the `threat_score` and `impersonation_score` features to near zero.

### Hardening Results
Two hardening strategies were applied:
1. **Adversarial Training**: Augmented the training dataset with a subset of adversarial samples.
2. **Ensemble Diversity**: Combined the outputs of the Classical Pipeline with the Transformer Model to flag disagreements.

**Post-Hardening Evasion Rate**: 12% (a 33% absolute reduction in evasion).
