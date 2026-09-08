import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

// Let transformers.js fetch the model from the HF Hub CDN (default) and cache
// it with the browser's Cache API, so repeat visits skip the download.
env.allowLocalModels = false;

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const CACHE_KEY = "wfyr_embeddings_v1";
const TOP_K = 5;

const statusText = document.getElementById("statusText");
const statusRow = document.getElementById("status");
const progressBar = document.getElementById("progressBar");
const qform = document.getElementById("qform");
const questionInput = document.getElementById("question");
const askBtn = document.getElementById("askBtn");
const resultsCard = document.getElementById("resultsCard");
const chunksEl = document.getElementById("chunks");
const genCard = document.getElementById("genCard");
const providerSelect = document.getElementById("provider");
const modelInput = document.getElementById("model");
const apiKeyInput = document.getElementById("apiKey");
const rememberKeyInput = document.getElementById("rememberKey");
const generateBtn = document.getElementById("generateBtn");
const genError = document.getElementById("genError");
const answerEl = document.getElementById("answer");

let extractor = null;
let chunks = [];
let vectors = []; // parallel array of Float32Array, normalized

function setStatus(text) {
  statusText.textContent = text;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

async function embed(text) {
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Float32Array.from(output.data);
}

function loadCachedVectors(expectedCount) {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length !== expectedCount) return null;
    return parsed.map((arr) => Float32Array.from(arr));
  } catch {
    return null;
  }
}

function saveCachedVectors(vecs) {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(vecs.map((v) => Array.from(v))));
  } catch {
    // Storage full or unavailable -- fine, we just re-embed next visit.
  }
}

async function init() {
  setStatus("Loading embedding model (first visit only, ~25MB)...");
  extractor = await pipeline("feature-extraction", MODEL_ID, {
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        const pct = Math.round((p.loaded / p.total) * 100);
        setStatus(`Downloading embedding model... ${pct}%`);
      }
    },
  });

  const resp = await fetch("data/chunks.json");
  chunks = await resp.json();

  const cached = loadCachedVectors(chunks.length);
  if (cached) {
    vectors = cached;
    setStatus(`Ready - ${chunks.length} passages indexed (cached).`);
  } else {
    setStatus(`Embedding ${chunks.length} passages in your browser (one-time, ~30-60s)...`);
    progressBar.hidden = false;
    progressBar.max = chunks.length;
    vectors = [];
    for (let i = 0; i < chunks.length; i++) {
      vectors.push(await embed(chunks[i].text));
      progressBar.value = i + 1;
      if (i % 10 === 0) setStatus(`Embedding passages... ${i + 1}/${chunks.length}`);
    }
    progressBar.hidden = true;
    saveCachedVectors(vectors);
    setStatus(`Ready - ${chunks.length} passages indexed.`);
  }

  statusRow.querySelector(".spinner").remove();
  questionInput.disabled = false;
  askBtn.disabled = false;

  const rememberedKey = localStorage.getItem("wfyr_api_key");
  if (rememberedKey) {
    apiKeyInput.value = rememberedKey;
    rememberKeyInput.checked = true;
  }
}

function renderChunks(results) {
  chunksEl.innerHTML = "";
  for (const { chunk, score } of results) {
    const div = document.createElement("div");
    div.className = "chunk";
    div.innerHTML = `
      <a class="chunk-source" href="${chunk.source_url}" target="_blank" rel="noopener">${chunk.source_title || chunk.source_url}</a>
      <span class="hint"> (similarity ${score.toFixed(2)})</span>
      <div class="chunk-text"></div>
    `;
    div.querySelector(".chunk-text").textContent = chunk.text;
    chunksEl.appendChild(div);
  }
  resultsCard.hidden = false;
  genCard.hidden = false;
  answerEl.textContent = "";
  genError.hidden = true;
}

let lastResults = [];

qform.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question || vectors.length === 0) return;

  askBtn.disabled = true;
  askBtn.textContent = "Searching...";
  const qVec = await embed(question);
  const scored = vectors.map((v, i) => ({ chunk: chunks[i], score: dot(qVec, v) }));
  scored.sort((a, b) => b.score - a.score);
  lastResults = scored.slice(0, TOP_K);
  renderChunks(lastResults);
  askBtn.disabled = false;
  askBtn.textContent = "Ask";
});

providerSelect.addEventListener("change", () => {
  modelInput.value = providerSelect.value === "anthropic" ? "claude-sonnet-5" : "";
  modelInput.placeholder = providerSelect.value === "openai" ? "e.g. gpt-4o-mini" : "";
});

function buildSystemPrompt() {
  return "You are a helpful assistant answering questions about being a first-year student at the University of Waterloo, based only on the provided context from uwaterloo.ca. If the context doesn't contain the answer, say so plainly instead of guessing. Cite the source title(s) you used at the end of your answer.";
}

function buildContext() {
  return lastResults
    .map(({ chunk }, i) => `[${i + 1}] Source: ${chunk.source_title} (${chunk.source_url})\n${chunk.text}`)
    .join("\n\n");
}

async function callAnthropic(apiKey, model, question) {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model,
      max_tokens: 600,
      system: buildSystemPrompt(),
      messages: [{ role: "user", content: `Context:\n\n${buildContext()}\n\nQuestion: ${question}` }],
    }),
  });
  if (!resp.ok) throw new Error(`Anthropic API error ${resp.status}: ${await resp.text()}`);
  const data = await resp.json();
  return data.content[0].text;
}

async function callOpenAI(apiKey, model, question) {
  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: buildSystemPrompt() },
        { role: "user", content: `Context:\n\n${buildContext()}\n\nQuestion: ${question}` },
      ],
    }),
  });
  if (!resp.ok) throw new Error(`OpenAI API error ${resp.status}: ${await resp.text()}`);
  const data = await resp.json();
  return data.choices[0].message.content;
}

generateBtn.addEventListener("click", async () => {
  const apiKey = apiKeyInput.value.trim();
  const model = modelInput.value.trim();
  const question = questionInput.value.trim();
  genError.hidden = true;
  answerEl.textContent = "";

  if (!apiKey || !question || lastResults.length === 0) {
    genError.textContent = "Ask a question first, then enter an API key.";
    genError.hidden = false;
    return;
  }

  if (rememberKeyInput.checked) {
    try { localStorage.setItem("wfyr_api_key", apiKey); } catch { /* ignore */ }
  } else {
    try { localStorage.removeItem("wfyr_api_key"); } catch { /* ignore */ }
  }

  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";
  try {
    const answer = providerSelect.value === "anthropic"
      ? await callAnthropic(apiKey, model, question)
      : await callOpenAI(apiKey, model, question);
    answerEl.textContent = answer;
  } catch (err) {
    genError.textContent = `${err.message} (if this looks like a CORS/network error, the provider may not allow direct browser calls -- try the other provider, or run rag/query.py locally instead).`;
    genError.hidden = false;
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Generate answer";
  }
});

init().catch((err) => {
  setStatus(`Failed to load: ${err.message}`);
  console.error(err);
});
