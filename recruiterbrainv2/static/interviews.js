const modeSelect = document.getElementById("analysis-mode");
const candidateField = document.querySelector('[data-mode="candidate"]');
const interviewField = document.querySelector('[data-mode="interview"]');
const jobField = document.querySelector('[data-mode="job"]');
const jdField = document.querySelector('[data-mode="jd"]');
const candidateInput = document.getElementById("candidate-id");
const interviewInput = document.getElementById("interview-id");
const jobInput = document.getElementById("job-id");
const jdInput = document.getElementById("jd-text");
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

function truncate(text, maxLength = 160) {
  if (!text) return "";
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 3)}...`;
}

function splitTags(value, limit = 8) {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value.filter(Boolean).slice(0, limit);
  }
  return value
    .split(/[,\n;/|]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, limit);
}

function updateFields() {
  const mode = modeSelect.value;
  candidateField.style.display = mode === "candidate" ? "block" : "none";
  interviewField.style.display = mode === "interview" ? "block" : "none";
  jobField.style.display = mode === "job" ? "block" : "none";
  jdField.style.display = mode === "jd" ? "block" : "none";
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
      <div class="summary-value">${summary.overall_score ?? "n/a"}</div>
    </div>
    <div class="summary-card">
      <div class="summary-label">Avg Sentiment</div>
      <div class="summary-value">${summary.avg_sentiment ?? "n/a"}</div>
    </div>
    <div class="summary-card">
      <div class="summary-label">Sentiment</div>
      <div class="summary-value">${summary.sentiment_label ?? "n/a"}</div>
    </div>
    <div class="summary-card">
      <div class="summary-label">Interviews</div>
      <div class="summary-value">${summary.interview_count ?? "n/a"}</div>
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
            Candidate: ${escapeHtml(item.candidate_id || "n/a")}
            ${item.job_id ? ` | Job: ${escapeHtml(item.job_id)}` : ""}
          </div>
          ${item.job_title ? `<div class="analysis-subtitle">Role: ${escapeHtml(item.job_title)}</div>` : ""}
          ${item.job_description ? `<div class="analysis-description">${escapeHtml(truncate(item.job_description))}</div>` : ""}
        </div>
        <div class="analysis-metrics">
          <span class="score-pill">Score ${item.overall_score ?? "n/a"}</span>
          <span class="sentiment-pill ${sentimentClass(item.sentiment_label)}">
            ${item.sentiment_label || "neutral"}
          </span>
        </div>
      </div>

      <div class="analysis-keywords">
        <div>
          <div class="keyword-label">Tech Keywords</div>
          <div class="keyword-list">
            ${tech.length ? tech.map((k) => `<span>${escapeHtml(k)}</span>`).join("") : "n/a"}
          </div>
        </div>
        <div>
          <div class="keyword-label">Top Keywords</div>
          <div class="keyword-list">
            ${top.length ? top.map((k) => `<span>${escapeHtml(k)}</span>`).join("") : "n/a"}
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
  const profile = candidate.candidate_profile || {};
  const name = profile.name || candidate.candidate_id || "Candidate";
  const location = [profile.location_city, profile.location_state, profile.location_country]
    .filter(Boolean)
    .join(", ");
  const titles = profile.top_3_titles;
  const summary = profile.semantic_summary ? truncate(profile.semantic_summary, 200) : "";
  const resumeSkills = splitTags(
    profile.skills_extracted || profile.tech_stack_primary || profile.tools_and_technologies,
    8
  );

  return `
    <div class="analysis-card">
      <div class="analysis-card-header">
        <div>
          <div class="analysis-title">${escapeHtml(name)}</div>
          <div class="analysis-subtitle">
            ID: ${escapeHtml(candidate.candidate_id || "n/a")} | Interviews: ${candidate.interview_count} | Answers: ${candidate.answer_count}
          </div>
          ${profile.career_stage ? `<div class="analysis-subtitle">Level: ${escapeHtml(profile.career_stage)}</div>` : ""}
          ${location ? `<div class="analysis-subtitle">Location: ${escapeHtml(location)}</div>` : ""}
          ${profile.total_experience_years ? `<div class="analysis-subtitle">Experience: ${escapeHtml(profile.total_experience_years.toString())} yrs</div>` : ""}
          ${titles ? `<div class="analysis-subtitle">Titles: ${escapeHtml(titles)}</div>` : ""}
          ${summary ? `<div class="analysis-description">${escapeHtml(summary)}</div>` : ""}
        </div>
        <div class="analysis-metrics">
          <span class="score-pill">Score ${candidate.overall_score ?? "n/a"}</span>
          <span class="sentiment-pill ${sentimentClass(candidate.sentiment_label)}">
            ${candidate.sentiment_label || "neutral"}
          </span>
        </div>
      </div>

      <div class="analysis-keywords">
        <div>
          <div class="keyword-label">Resume Skills</div>
          <div class="keyword-list">
            ${resumeSkills.length ? resumeSkills.map((k) => `<span>${escapeHtml(k)}</span>`).join("") : "n/a"}
          </div>
        </div>
        <div>
          <div class="keyword-label">Tech Keywords</div>
          <div class="keyword-list">
            ${(candidate.tech_keywords || []).slice(0, 8).map((k) => `<span>${escapeHtml(k)}</span>`).join("") || "n/a"}
          </div>
        </div>
        <div>
          <div class="keyword-label">Top Keywords</div>
          <div class="keyword-list">
            ${(candidate.top_keywords || []).slice(0, 8).map((k) => `<span>${escapeHtml(k)}</span>`).join("") || "n/a"}
          </div>
        </div>
      </div>

      <button class="btn-secondary analyze-candidate-btn" data-candidate="${escapeHtml(candidate.candidate_id || "")}">
        Analyze Candidate Interviews
      </button>
    </div>
  `;
}

function renderJdCandidateCard(candidate) {
  const profile = candidate.candidate_profile || {};
  const name = profile.name || candidate.candidate_id || "Candidate";
  const location = [profile.location_city, profile.location_state, profile.location_country]
    .filter(Boolean)
    .join(", ");
  const titles = profile.top_3_titles;
  const summary = profile.semantic_summary ? truncate(profile.semantic_summary, 200) : "";
  const matched = candidate.matched_skills || [];
  const missing = candidate.missing_skills || [];
  const nice = candidate.nice_to_have_matched || [];
  const resumeOnly = candidate.resume_only_skills || [];
  const evidence = candidate.evidence || {};

  const evidenceBlocks = Object.entries(evidence).map(([skill, snippets]) => {
    const snippetHtml = (snippets || [])
      .map(
        (s) => `
          <div class="answer-row">
            <div class="answer-meta">Skill: ${escapeHtml(skill)} | Score ${s.quality_score ?? "n/a"}</div>
            <div class="answer-text">${escapeHtml(s.snippet || "")}</div>
          </div>
        `
      )
      .join("");
    return `
      <div class="analysis-section">
        <div class="keyword-label">Evidence: ${escapeHtml(skill)}</div>
        ${snippetHtml || "<div class=\"analysis-subtitle\">n/a</div>"}
      </div>
    `;
  });

  return `
    <div class="analysis-card">
      <div class="analysis-card-header">
        <div>
          <div class="analysis-title">${escapeHtml(name)}</div>
          <div class="analysis-subtitle">
            ID: ${escapeHtml(candidate.candidate_id || "n/a")} | Score ${candidate.overall_score ?? "n/a"}
          </div>
          ${profile.career_stage ? `<div class="analysis-subtitle">Level: ${escapeHtml(profile.career_stage)}</div>` : ""}
          ${location ? `<div class="analysis-subtitle">Location: ${escapeHtml(location)}</div>` : ""}
          ${profile.total_experience_years ? `<div class="analysis-subtitle">Experience: ${escapeHtml(profile.total_experience_years.toString())} yrs</div>` : ""}
          ${titles ? `<div class="analysis-subtitle">Titles: ${escapeHtml(titles)}</div>` : ""}
          ${summary ? `<div class="analysis-description">${escapeHtml(summary)}</div>` : ""}
        </div>
        <div class="analysis-metrics">
          <span class="score-pill">Coverage ${(candidate.coverage_ratio ?? 0) * 100}%</span>
          <span class="sentiment-pill sentiment-neutral">Depth ${(candidate.depth_score ?? 0) * 100}%</span>
        </div>
      </div>

      <div class="analysis-keywords">
        <div>
          <div class="keyword-label">Matched Skills</div>
          <div class="keyword-list">
            ${matched.length ? matched.map((k) => `<span>${escapeHtml(k)}</span>`).join("") : "n/a"}
          </div>
        </div>
        <div>
          <div class="keyword-label">Missing Skills</div>
          <div class="keyword-list">
            ${missing.length ? missing.map((k) => `<span>${escapeHtml(k)}</span>`).join("") : "n/a"}
          </div>
        </div>
      </div>

      <div class="analysis-keywords">
        <div>
          <div class="keyword-label">Nice-to-Have</div>
          <div class="keyword-list">
            ${nice.length ? nice.map((k) => `<span>${escapeHtml(k)}</span>`).join("") : "n/a"}
          </div>
        </div>
        <div>
          <div class="keyword-label">Resume-Only Skills</div>
          <div class="keyword-list">
            ${resumeOnly.length ? resumeOnly.map((k) => `<span>${escapeHtml(k)}</span>`).join("") : "n/a"}
          </div>
        </div>
      </div>

      ${evidenceBlocks.join("") || ""}
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
          <div class="summary-value">${escapeHtml(data.job_id || "n/a")}</div>
        </div>
        <div class="summary-card">
          <div class="summary-label">Job Title</div>
          <div class="summary-value">${escapeHtml(data.job_title || "n/a")}</div>
        </div>
      </div>
      ${data.job_description ? `<div class="analysis-description">${escapeHtml(truncate(data.job_description, 220))}</div>` : ""}
    `;
    resultsDiv.innerHTML = (data.candidates || []).map(renderCandidateCard).join("");
    return;
  }

  if (data.mode === "jd") {
    const jd = data.jd_summary || {};
    summaryDiv.innerHTML = `
      <div class="summary-grid">
        <div class="summary-card">
          <div class="summary-label">Candidates</div>
          <div class="summary-value">${data.total_candidates ?? 0}</div>
        </div>
        <div class="summary-card">
          <div class="summary-label">Seniority</div>
          <div class="summary-value">${escapeHtml(jd.seniority_level || "Any")}</div>
        </div>
      </div>
      <div class="analysis-keywords" style="margin-top: 1rem;">
        <div>
          <div class="keyword-label">Must Have</div>
          <div class="keyword-list">
            ${(jd.must_have_skills || []).map((k) => `<span>${escapeHtml(k)}</span>`).join("") || "n/a"}
          </div>
        </div>
        <div>
          <div class="keyword-label">Nice to Have</div>
          <div class="keyword-list">
            ${(jd.nice_to_have_skills || []).map((k) => `<span>${escapeHtml(k)}</span>`).join("") || "n/a"}
          </div>
        </div>
        <div>
          <div class="keyword-label">Domain</div>
          <div class="keyword-list">
            ${(jd.domain_keywords || []).map((k) => `<span>${escapeHtml(k)}</span>`).join("") || "n/a"}
          </div>
        </div>
      </div>
    `;
    resultsDiv.innerHTML = (data.candidates || []).map(renderJdCandidateCard).join("");
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
  } else if (mode === "jd") {
    payload.jd_text = jdInput.value.trim();
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
