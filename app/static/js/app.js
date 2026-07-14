const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const els = {
  text: $('#review-text'), count: $('#char-count'), alert: $('#form-alert'), analyze: $('#analyze'),
  clear: $('#clear-text'), reset: $('#reset-defaults'), result: $('#result'), explanation: $('#reasoner-result'),
  classificationResult: $('#classification-result'), entityResult: $('#entity-result'),
  classifier: $('#classifier-threshold'), ner: $('#ner-threshold'), temperature: $('#temperature'),
  classifierValue: $('#classifier-value'), nerValue: $('#ner-value'), temperatureValue: $('#temperature-value'),
  header: $('#site-header'), copy: $('#copy-explanation'), outputStatus: $('#output-status'),
  ocrInput: $('#ocr-image'), ocrPicker: $('#ocr-picker'), ocrButton: $('#ocr-analyze'),
  ocrPreview: $('#ocr-preview'), ocrFileName: $('#ocr-file-name'),
};
let defaults;
let explanationText = '';
let previousScroll = 0;
let selectedImage = null;
let previewUrl = null;
const entityColors = ['#62a8ff', '#a98cff', '#ef78ac', '#51c7ae', '#e5a14b', '#8592ff', '#e46f60', '#4eb5d1'];
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const percentage = value => `${(Number(value || 0) * 100).toFixed(1)}%`;
const displayLabel = value => String(value || '').replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase());
const colorFor = label => entityColors[Math.abs([...String(label)].reduce((sum, character) => ((sum * 31) + character.charCodeAt(0)) | 0, 7)) % entityColors.length];

const syncControls = () => {
  els.classifierValue.textContent = Number(els.classifier.value).toFixed(2);
  els.nerValue.textContent = Number(els.ner.value).toFixed(2);
  els.temperatureValue.textContent = Number(els.temperature.value).toFixed(2);
};

const updateCount = () => {
  els.count.textContent = `${els.text.value.length.toLocaleString()} / ${Number(els.text.maxLength).toLocaleString()}`;
};

async function loadDefaults() {
  try {
    const response = await fetch('/api/defaults');
    if (!response.ok) throw new Error();
    defaults = await response.json();
    applyDefaults();
  } catch { showAlert('Model settings are temporarily unavailable.'); }
}

function applyDefaults() {
  if (!defaults) return;
  els.classifier.value = defaults.classifier_threshold;
  els.ner.value = defaults.ner_threshold;
  els.temperature.value = defaults.temperature;
  els.text.maxLength = defaults.max_chars;
  syncControls(); updateCount();
}

function showAlert(message) { els.alert.textContent = message; els.alert.hidden = false; }
function hideAlert() { els.alert.hidden = true; els.alert.textContent = ''; }

async function analyze() {
  const text = els.text.value.trim();
  if (!text) { showAlert('Add a review to continue.'); els.text.focus(); return; }
  hideAlert();
  els.analyze.disabled = true;
  els.analyze.setAttribute('aria-busy', 'true');
  els.outputStatus.classList.add('running');
  els.outputStatus.innerHTML = '<i></i> Running';
  ['classification', 'entity', 'explanation'].forEach(stage => $(`#${stage}-stage`).classList.add('running'));
  $('#classification-state').textContent = 'Running';
  $('#entity-state').textContent = 'Queued';
  els.copy.disabled = true;
  try {
    const response = await fetch('/api/analyze', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        text, classifier_threshold: Number(els.classifier.value),
        ner_threshold: Number(els.ner.value), temperature: Number(els.temperature.value),
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(formatError(payload.detail));
    renderResult(payload);
  } catch (error) {
    showAlert(error.message || 'The analysis could not be completed.');
    els.outputStatus.innerHTML = '<i></i> Error';
  } finally {
    els.analyze.disabled = false;
    els.analyze.removeAttribute('aria-busy');
    els.outputStatus.classList.remove('running');
    ['classification', 'entity', 'explanation'].forEach(stage => $(`#${stage}-stage`).classList.remove('running'));
  }
}

async function analyzeImage() {
  if (!selectedImage) return;
  hideAlert();
  els.ocrButton.disabled = true;
  els.ocrButton.setAttribute('aria-busy', 'true');
  els.analyze.disabled = true;
  els.outputStatus.classList.add('running');
  els.outputStatus.innerHTML = '<i></i> OCR / Running';
  ['classification', 'entity', 'explanation'].forEach(stage => $(`#${stage}-stage`).classList.add('running'));

  const form = new FormData();
  form.append('image', selectedImage);
  form.append('classifier_threshold', els.classifier.value);
  form.append('ner_threshold', els.ner.value);
  form.append('temperature', els.temperature.value);
  try {
    const response = await fetch('/api/analyze-image', {method: 'POST', body: form});
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(formatError(payload.detail));
    els.text.value = payload.input_text || payload.ocr?.text || '';
    updateCount();
    els.ocrFileName.textContent = 'Text extracted successfully';
    renderResult(payload);
  } catch (error) {
    showAlert(error.message || 'The image could not be analyzed.');
    els.outputStatus.innerHTML = '<i></i> OCR error';
  } finally {
    els.ocrButton.disabled = !selectedImage;
    els.ocrButton.removeAttribute('aria-busy');
    els.analyze.disabled = false;
    els.outputStatus.classList.remove('running');
    ['classification', 'entity', 'explanation'].forEach(stage => $(`#${stage}-stage`).classList.remove('running'));
  }
}

function selectImage(file) {
  if (!file) return;
  selectedImage = file;
  els.ocrButton.disabled = false;
  els.ocrFileName.textContent = file.name;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  els.ocrPreview.replaceChildren();
  const preview = document.createElement('img');
  preview.src = previewUrl;
  preview.alt = '';
  els.ocrPreview.append(preview);
}

function formatError(detail) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map(item => item.msg).join(' ');
  return 'The analysis could not be completed.';
}

function renderResult(payload) {
  renderClassification(payload.classifier);
  renderEntities(payload.input_text, payload.ner);
  els.explanation.innerHTML = payload.reasoner?.html || '<p>No explanation was returned.</p>';
  explanationText = els.explanation.innerText.trim();
  els.copy.disabled = false;
  els.outputStatus.classList.remove('running');
  els.outputStatus.innerHTML = '<i></i> Complete';
  if (window.innerWidth <= 1000) {
    setTimeout(() => els.result.scrollIntoView({behavior: 'smooth', block: 'start'}), 120);
  } else {
    els.result.scrollTo({top: 0, behavior: 'smooth'});
  }
}

function renderClassification(classifier) {
  const calibrated = classifier?.calibrated || {};
  const selective = classifier?.selective_prediction || {};
  const accepted = Boolean(selective.accepted);
  const probabilities = Object.entries(calibrated.probabilities || {}).map(([label, probability]) => `
    <div class="confidence-row">
      <span>${escapeHtml(displayLabel(label))}</span>
      <div class="confidence-track"><div class="confidence-fill" style="width:${Math.max(0, Math.min(100, Number(probability) * 100))}%"></div></div>
      <b>${percentage(probability)}</b>
    </div>`).join('');
  els.classificationResult.innerHTML = `
    <div class="classification-summary">
      <div><div class="classification-label">${escapeHtml(displayLabel(calibrated.label || 'Unknown'))}</div><p class="threshold-note">Top confidence ${percentage(calibrated.confidence)}</p></div>
      <div class="acceptance ${accepted ? '' : 'rejected'}"><i></i>${accepted ? 'Accepted prediction' : 'Not accepted — uncertain'}</div>
    </div>
    <div class="confidence-list">${probabilities}</div>
    <p class="threshold-note">Acceptance threshold ${percentage(selective.confidence_threshold)}</p>`;
  $('#classification-state').textContent = accepted ? 'Accepted' : 'Uncertain';
}

function renderEntities(text, ner) {
  const entities = ner?.entities || [];
  if (!entities.length) {
    els.entityResult.innerHTML = '<p class="empty-entities">No words were predicted above the selected entity threshold.</p>';
    $('#entity-state').textContent = 'None found';
    return;
  }
  const annotated = (ner.segments || []).map(segment => {
    if (!segment.entities?.length) return escapeHtml(segment.text);
    const primary = segment.entities[0];
    const tooltip = segment.entities.map(entity => `${displayLabel(entity.label)} · ${percentage(entity.confidence)}`).join(' | ');
    return `<span class="entity-highlight" tabindex="0" style="--entity-color:${colorFor(primary.label)}" data-tooltip="${escapeHtml(tooltip)}" aria-label="${escapeHtml(segment.text)}: ${escapeHtml(tooltip)}">${escapeHtml(segment.text)}</span>`;
  }).join('');
  const key = entities.map(entity => `<span style="--entity-color:${colorFor(entity.label)}"><i></i>${escapeHtml(entity.text)} · ${percentage(entity.confidence)}</span>`).join('');
  els.entityResult.innerHTML = `<div class="annotated-review">${annotated}</div><div class="entity-key">${key}</div>`;
  $('#entity-state').textContent = `${entities.length} predicted`;
}

async function copyExplanation() {
  try {
    await navigator.clipboard.writeText(explanationText);
    els.copy.textContent = 'Copied';
    setTimeout(() => { els.copy.textContent = 'Copy explanation'; }, 1400);
  } catch { showAlert('Clipboard access is unavailable.'); }
}

function handleScroll() {
  const current = window.scrollY;
  els.header.classList.toggle('scrolled', current > 20);
  els.header.classList.toggle('hidden', current > previousScroll && current > 130);
  previousScroll = current;
}

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => { if (entry.isIntersecting) { entry.target.classList.add('visible'); observer.unobserve(entry.target); } });
}, {threshold: .12});

window.addEventListener('load', () => {
  setTimeout(() => { document.body.classList.remove('is-loading'); document.body.classList.add('loaded'); }, 350);
});
window.addEventListener('scroll', handleScroll, {passive: true});
$$('.reveal').forEach(element => observer.observe(element));
[els.classifier, els.ner, els.temperature].forEach(control => control.addEventListener('input', syncControls));
els.text.addEventListener('input', updateCount);
els.text.addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') analyze(); });
els.analyze.addEventListener('click', analyze);
els.clear.addEventListener('click', () => { els.text.value = ''; updateCount(); hideAlert(); els.text.focus(); });
els.reset.addEventListener('click', applyDefaults);
els.copy.addEventListener('click', copyExplanation);
els.ocrPicker.addEventListener('click', () => els.ocrInput.click());
els.ocrInput.addEventListener('change', () => selectImage(els.ocrInput.files?.[0]));
els.ocrButton.addEventListener('click', analyzeImage);

syncControls(); updateCount(); loadDefaults(); handleScroll();
