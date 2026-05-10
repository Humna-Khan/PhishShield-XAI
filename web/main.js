import './style.css'

const apiBase = 'http://localhost:8000';

// Elements
const emailInput = document.getElementById('email-input');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const resultsSection = document.getElementById('results');
const apiStatus = document.getElementById('api-status');

// Status Check
async function checkStatus() {
  try {
    const res = await fetch(`${apiBase}/health`);
    const data = await res.json();
    if (data.status === 'healthy') {
      apiStatus.classList.add('online');
      apiStatus.classList.remove('offline');
      apiStatus.innerHTML = '<span class="dot"></span> API Online';
    }
  } catch (err) {
    apiStatus.classList.add('offline');
    apiStatus.classList.remove('online');
    apiStatus.innerHTML = '<span class="dot"></span> API Offline';
  }
}

// Analysis
async function runAnalysis() {
  const text = emailInput.value.trim();
  if (!text) return alert('Please enter email text');

  analyzeBtn.classList.add('loading');
  analyzeBtn.disabled = true;

  try {
    const res = await fetch(`${apiBase}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email_text: text })
    });

    if (!res.ok) throw new Error('API Error');

    const data = await res.json();
    displayResults(data);
  } catch (err) {
    alert('Failed to connect to API. Is it running?');
  } finally {
    analyzeBtn.classList.remove('loading');
    analyzeBtn.disabled = false;
  }
}

function displayResults(data) {
  resultsSection.classList.remove('hidden');
  resultsSection.scrollIntoView({ behavior: 'smooth' });

  // Classification
  const predText = document.getElementById('prediction-text');
  const confBar = document.getElementById('confidence-bar');
  const confScore = document.getElementById('confidence-score');
  
  predText.innerText = data.classification.toUpperCase();
  predText.className = `prediction-value ${data.classification}`;
  
  const percentage = (data.confidence_score * 100).toFixed(1);
  confBar.style.width = `${percentage}%`;
  confScore.innerText = `${percentage}%`;
  confBar.style.backgroundColor = data.classification === 'phishing' ? 'var(--phishing)' : 'var(--legit)';

  // LIME Highlights
  const limeContainer = document.getElementById('lime-highlights');
  limeContainer.innerHTML = '';
  data.lime_highlights.forEach(h => {
    if (h.token === 'N/A') return;
    const span = document.createElement('span');
    span.className = `token ${h.weight > 0 ? 'pos' : h.weight < 0 ? 'neg' : ''}`;
    span.innerText = h.token;
    span.title = `Weight: ${h.weight.toFixed(4)}`;
    limeContainer.appendChild(span);
  });

  // Explanation
  document.getElementById('llm-explanation').innerText = data.llm_explanation;

  // SHAP
  const shapList = document.getElementById('shap-list');
  shapList.innerHTML = '';
  data.shap_features.forEach(f => {
    const item = document.createElement('div');
    item.className = 'shap-item';
    item.innerHTML = `
      <span class="feat-name">${f.feature}</span>
      <span class="feat-val" style="color: ${f.direction === 'phishing' ? 'var(--phishing)' : 'var(--legit)'}">
        ${f.importance.toFixed(4)}
      </span>
    `;
    shapList.appendChild(item);
  });
}

// Events
analyzeBtn.addEventListener('click', runAnalysis);
clearBtn.addEventListener('click', () => {
  emailInput.value = '';
  resultsSection.classList.add('hidden');
});

// Init
checkStatus();
setInterval(checkStatus, 5000);
