const reviewText = document.querySelector('#reviewText');
const signalList = document.querySelector('#signalList');
const wordCount = document.querySelector('#wordCount');
const saveState = document.querySelector('#saveState');
const statusText = document.querySelector('#statusText');
const goButton = document.querySelector('#goButton');
const reviewId = crypto.randomUUID();

const labels = { food_quality: 'Food quality', ambience: 'Ambience', prices: 'Prices', location: 'Location', general: 'General' };

function renderEmpty() {
  signalList.innerHTML = '<div class="empty-state"><span class="empty-icon">✦</span><p>Your aspect signals<br>will appear here.</p></div>';
}

function renderSignals(results) {
  signalList.innerHTML = Object.entries(results).map(([key, result]) => `
    <article class="signal ${result.sentiment}">
      <div class="signal-heading"><span>${labels[key]}</span><span class="signal-sentiment">${result.sentiment}</span></div>
      <div class="stars" aria-label="${result.stars} out of 5 stars">${'★'.repeat(result.stars)}<span>${'★'.repeat(5 - result.stars)}</span></div>
      <div class="signal-meta"><span>${result.matched_terms.length ? `matched: ${result.matched_terms.slice(0, 2).join(', ')}` : 'awaiting aspect evidence'}</span><strong>${result.stars}/5</strong></div>
    </article>`).join('');
}

async function scoreReview() {
  const text = reviewText.value.trim();
  wordCount.textContent = `${text ? text.split(/\s+/).length : 0} words`;
  if (!text) {
    renderEmpty();
    saveState.textContent = 'Waiting for your review';
    return;
  }
  goButton.disabled = true;
  saveState.textContent = 'Scoring and saving...';
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ review_id: reviewId, text }) });
    if (!response.ok) throw new Error('Request failed');
    const payload = await response.json();
    renderSignals(payload.results);
    saveState.textContent = 'Saved just now';
    statusText.textContent = 'Baseline model online';
  } catch (error) {
    saveState.textContent = 'Could not reach the backend';
    statusText.textContent = 'Backend offline';
  } finally {
    goButton.disabled = false;
  }
}

reviewText.addEventListener('input', () => {
  const text = reviewText.value.trim();
  wordCount.textContent = `${text ? text.split(/\s+/).length : 0} words`;
  renderEmpty();
  saveState.textContent = text ? 'Draft ready — press Go' : 'Waiting for your review';
});
reviewText.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') scoreReview();
});
goButton.addEventListener('click', scoreReview);
renderEmpty();