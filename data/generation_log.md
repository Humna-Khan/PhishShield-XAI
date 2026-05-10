# Adversarial Phishing Email Generation — Methodology Documentation

## Overview

This document describes the methodology used to generate **500+ synthetic adversarial phishing emails** designed to evade detection by machine learning classifiers. These emails form a critical component of the adversarial testing and hardening pipeline.

## Generation Method

### Approach: Template-Based LLM Generation with Evasion Optimization

We use a hybrid approach combining **template engineering** with **LLM text generation** to produce phishing emails that:
1. Mimic natural human writing style (low perplexity)
2. Deliberately avoid classic phishing indicators
3. Embed malicious intent within legitimate-looking business contexts

### Generation Pipeline

```
1. Define Phishing Scenarios (10 categories)
   ↓
2. Create Base Templates per Scenario
   ↓
3. LLM Paraphrasing + Style Transfer
   ↓
4. Evasion Feature Injection
   ↓
5. Quality Filtering (perplexity + readability)
   ↓
6. Adversarial Validation (test against baseline model)
```

### Phishing Scenario Categories

| # | Category | Description | Count |
|---|----------|-------------|-------|
| 1 | **Invoice Fraud** | Fake invoices from "legitimate" vendors | 50 |
| 2 | **Credential Harvesting** | Account verification disguised as IT support | 60 |
| 3 | **CEO/BEC Fraud** | Impersonation of executives requesting wire transfers | 50 |
| 4 | **Document Sharing** | Malicious links disguised as shared documents | 50 |
| 5 | **Delivery Notification** | Fake shipping/delivery notifications | 50 |
| 6 | **HR/Benefits** | Fake HR communications about benefits/payroll | 50 |
| 7 | **Tech Support** | Social engineering via fake tech support | 50 |
| 8 | **Meeting/Calendar** | Malicious meeting invitations | 40 |
| 9 | **Legal/Compliance** | Fake compliance or legal notices | 50 |
| 10 | **Subscription Renewal** | Fake service renewal notices | 50 |

**Total: 500+ emails**

### Evasion Techniques Applied

These adversarial emails differ from standard AI-generated phishing in several key ways:

1. **No Urgency Overload**: Standard phishing uses excessive urgency words ("IMMEDIATELY", "URGENT"). Our adversarial samples use subtle, professional urgency ("at your earliest convenience", "by end of business").

2. **Contextual Coherence**: Each email includes realistic business context — proper company names, plausible sender roles, consistent email threads.

3. **Linguistic Naturalness**: Sentences follow natural English patterns with varied length, proper grammar, and no ALL-CAPS or excessive punctuation.

4. **Clean URLs**: Instead of suspicious shortened URLs, emails reference plausible-looking domains or internal file shares.

5. **Personalization**: Emails include realistic names, departments, and reference to previous communications.

6. **Professional Tone**: Emails match the tone of legitimate business correspondence — formal but not robotic.

### How These Differ from Standard AI-Generated Text

| Feature | Standard AI Phishing | Our Adversarial Samples |
|---------|---------------------|------------------------|
| Perplexity | High (robotic patterns) | Low (human-like) |
| Urgency Words | 5-10 per email | 0-2 per email |
| URL Suspiciousness | Shortened/random URLs | Clean domain names |
| Grammar Errors | Frequent | None |
| Personalization | Generic ("Dear User") | Specific ("Hi Sarah") |
| Business Context | Absent or vague | Detailed and coherent |
| Emotional Manipulation | Overt threats | Subtle social proof |
| Length | Short, template-like | Varied, natural |

## Quality Assurance

Each generated email passes through:
1. **Readability Check**: Flesch-Kincaid score between 30-70 (professional range)
2. **Perplexity Filter**: GPT-2 perplexity below threshold (filters robotic text)
3. **Deduplication**: Cosine similarity check against existing corpus
4. **Manual Review**: 10% random sample reviewed for quality

## Ethical Considerations

- All emails are synthetic — no real PII or organizations
- Generated for research purposes only
- Dataset will be clearly labeled as adversarial/synthetic
- Not distributed outside academic context
