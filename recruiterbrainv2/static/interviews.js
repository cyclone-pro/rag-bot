const modeSelect = document.getElementById("analysis-mode");
const candidateField = document.querySelector('[data-mode="candidate"]');
const interviewField = document.querySelector('[data-mode="interview"]');
const jobField = document.querySelector('[data-mode="job"]');
const candidateInput = document.getElementById("candidate-id");
const interviewInput = document.getElementById("interview-id");
const jobInput = document.getElementById("job-id");
const latestOnly = document.getElementById("latest-only");
const limitInput = document.getElementById("record-limit");
const analyzeBtn = document.getElementById("analyze-btn");
const statusDiv = document.getElementById("analysis-status");
const summaryDiv = document.getElementById("analysis-summary");
const resultsDiv = document.getElementById("analysis-results");

function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function updateFields() {
  const mode = modeSelect.value;
  candidateField.style.display = mode === "candidate" ? "block" : "none";
  interviewField.style.display = mode === "interview" ? "block" : "none";
  jobField.style.display = mode === "job" ? "block" : "none";
}

function setStatus(message, type = "info") {
  statusDiv.textContent = message;
  statusDiv.className = `analysis-status ${type}`;
}

function sentimentClass(label) {
  const map = {
    very_positive: "sentiment-positive",
    positive: "sentiment-positive",
    neutral: "sentiment-neutral",
    negative: "sentiment-negative",
    very_negative: "sentiment-negative",
  };
  return map[label] || "sentiment-neutral";
}

function renderSummaryCards(summary) {
  if (!summary) {
    return "No summary available.";
  }
  return `
    <div class="summary-card">
      <div class="summary-label">Overall Score</div>
      <div class="summary-value">${summary.overall_score ?? "—"}</div>
    </div>
    <div class="summary-card">
      <div class="summary-label">Avg Sentiment</div>
      <div class="summary-value">${summary.avg_sentiment ?? "—"}</div>
    </div>
    <div class="summary-card">
      <div class="summary-label">Sentiment</div>
      <div class="summary-value">${summary.sentiment_label ?? "—"}</div>
    </div>
    <div class="summary-card">
      <div class="summary-label">Interviews</div>
      <div class="summary-value">${summary.interview_count ?? "—"}</div>
    </div>
  `;
}

function renderInterviewCard(item) {
  const keywords = item.keywords || {};
  const tech = (keywords.tech_keywords || []).slice(0, 8);
  const top = (keywords.all_keywords || []).slice(0, 8);
  const answers = item.answers || [];

  return `
    <div class="analysis-card">
      <div class="analysis-card-header">
        <div>
          <div class="analysis-title">${escapeHtml(item.interview_id || "Interview")}</div>
          <div class="analysis-subtitle">
            Candidate: ${escapeHtml(item.candidate_id || "—")}
            ${item.job_id ? ` • Job: ${escapeHtml(item.job_id)}` : ""}
          </div>
        </div>
        <div class="analysis-metrics">
          <span class="score-pill">Score ${item.overall_score ?? "—"}</span>
          <span class="sentiment-pill ${sentimentClass(item.sentiment_label)}">
            ${item.sentiment_label || "neutral"}
          </span>
        </div>
      </div>

      <div class="analysis-keywords">
        <div>
          <div class="keyword-label">Tech Keywords</div>
          <div class="keyword-list">
            ${tech.length ? tech.map((k) => `<span>${escapeHtml(k)}</span>`).join("") : "—"}
          </div>
        </div>
        <div>
          <div class="keyword-label">Top Keywords</div>
          <div class="keyword-list">
            ${top.length ? top.map((k) => `<span>${escapeHtml(k)}</span>`).join("") : "—"}
          </div>
        </div>
      </div>

      <div class="analysis-answers">
        ${answers.map((ans) => `
          <div class="answer-row">
            <div class="answer-meta">
              Q${ans.question_index ?? "?"}
              <span class="sentiment-pill ${sentimentClass(ans.sentiment_label)}">
                ${ans.sentiment_label || "neutral"}
              </span>
            </div>
            <div class="answer-text">${escapeHtml(ans.answer_snippet || "")}</div>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderCandidateCard(candidate) {
  return `
    <div class="analysis-card">
      <div class="analysis-card-header">
        <div>
          <div class="analysis-title">${escapeHtml(candidate.candidate_id || "Candidate")}</div>
          <div class="analysis-subtitle">
            Interviews: ${candidate.interview_count} • Answers: ${candidate.answer_count}
          </div>
        </div>
        <div class="analysis-metrics">
          <span class="score-pill">Score ${candidate.overall_score ?? "—"}</span>
          <span class="sentiment-pill ${sentimentClass(candidate.sentiment_label)}">
            ${candidate.sentiment_label || "neutral"}
          </span>
        </div>
      </div>

      <div class="analysis-keywords">
        <div>
          <div class="keyword-label">Tech Keywords</div>
          <div class="keyword-list">
            ${(candidate.tech_keywords || []).slice(0, 8).map((k) => `<span>${escapeHtml(k)}</span>`).join("") || "—"}
          </div>
        </div>
        <div>
          <div class="keyword-label">Top Keywords</div>
          <div class="keyword-list">
            ${(candidate.top_keywords || []).slice(0, 8).map((k) => `<span>${escapeHtml(k)}</span>`).join("") || "—"}
          </div>
        </div>
      </div>

      <button class="btn-secondary analyze-candidate-btn" data-candidate="${escapeHtml(candidate.candidate_id || "")}">
        Analyze Candidate Interviews
      </button>
    </div>
  `;
}

function renderResults(data) {
  if (data.error) {
    setStatus(data.error, "error");
    summaryDiv.textContent = "Unable to generate analysis.";
    resultsDiv.innerHTML = "";
    return;
  }

  setStatus("Analysis complete.", "success");

  if (data.mode === "job") {
    summaryDiv.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card">
          <div class="summary-label">Candidates</div>
          <div class="summary-value">${data.total_candidates ?? 0}</div>
        </div>
        <div class="summary-card">
          <div class="summary-label">Job ID</div>
          <div class="summary-value">${escapeHtml(data.job_id || "—")}</div>
        </div>
      </div>
    `;
    resultsDiv.innerHTML = (data.candidates || []).map(renderCandidateCard).join("");
    return;
  }

  summaryDiv.innerHTML = `<div class="summary-grid">${renderSummaryCards(data.summary)}</div>`;
  resultsDiv.innerHTML = (data.interviews || []).map(renderInterviewCard).join("");
}

async function runAnalysis() {
  const mode = modeSelect.value;
  const payload = {
    mode,
    latest_only: latestOnly.checked,
    limit: parseInt(limitInput.value, 10) || 500,
  };

  if (mode === "candidate") {
    payload.candidate_id = candidateInput.value.trim();
  } else if (mode === "interview") {
    payload.interview_id = interviewInput.value.trim();
  } else if (mode === "job") {
    payload.job_id = jobInput.value.trim();
  }

  setStatus("Running analysis...", "info");
  analyzeBtn.disabled = true;
  resultsDiv.innerHTML = "";
  summaryDiv.textContent = "Working on it...";

  try {
    const response = await fetch("/v2/analyze_interviews", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    renderResults(data);
  } catch (error) {
    console.error("Interview analysis failed:", error);
    setStatus("Analysis failed. Please try again.", "error");
    summaryDiv.textContent = "Unable to load analysis.";
    resultsDiv.innerHTML = "";
  } finally {
    analyzeBtn.disabled = false;
  }
}

modeSelect.addEventListener("change", updateFields);
analyzeBtn.addEventListener("click", runAnalysis);

resultsDiv.addEventListener("click", (event) => {
  const btn = event.target.closest(".analyze-candidate-btn");
  if (!btn) return;
  const candidateId = btn.getAttribute("data-candidate");
  if (!candidateId) return;
  modeSelect.value = "candidate";
  updateFields();
  candidateInput.value = candidateId;
  runAnalysis();
});

updateFields();
