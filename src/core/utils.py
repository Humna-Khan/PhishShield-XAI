"""
Feature extraction utilities for the Phishing Detection API.
"""
import re
import numpy as np

def extract_features(text):
    text_lower = text.lower()
    words = text_lower.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    urgency = ['urgent','immediately','asap','now','hurry','expire','suspended',
               'verify','confirm','alert','warning']
    money = ['bank','account','password','credit','ssn','wire','transfer',
             'payment','invoice','bitcoin','wallet']
    threats = ['suspend','terminate','close','block','disable','unauthorized',
               'breach','compromise','violation','penalty']
    polite = ['dear','please','kindly','sir','madam','respected']
    impersonate = ['official','authorized','government','irs','microsoft',
                   'google','apple','amazon','paypal']

    return {
        'urgency_word_count': sum(1 for w in urgency if w in text_lower),
        'url_count': len(re.findall(r'https?://\S+', text)),
        'suspicious_url_count': len(re.findall(r'https?://(?:bit\.ly|tinyurl|goo\.gl)', text_lower)),
        'email_count': len(re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)),
        'has_html': 1 if re.search(r'<[^>]+>', text) else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'money_word_count': sum(1 for w in money if w in text_lower),
        'word_count': len(words),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'sentence_count': max(len(sentences), 1),
        'avg_sentence_length': len(words) / max(len(sentences), 1),
        'capital_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1),
        'politeness_score': sum(1 for w in polite if w in text_lower),
        'impersonation_score': sum(1 for w in impersonate if w in text_lower),
        'threat_score': sum(1 for w in threats if w in text_lower),
    }

def prepare_full_features(text, vectorizer):
    ling = extract_features(text)
    ling_values = np.array(list(ling.values())).reshape(1, -1)
    tfidf = vectorizer.transform([text]).toarray()
    features = np.hstack([ling_values, tfidf])
    names = list(ling.keys()) + [f"tfidf_{n}" for n in vectorizer.get_feature_names_out()]
    return features, names

def preprocess_text(text):
    """Clean and normalize email text."""
    import re
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text
