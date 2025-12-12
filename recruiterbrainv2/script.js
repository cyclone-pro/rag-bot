// ==================== STATE ====================
const state = {
  insightView: false,
  showContacts: false,
  isRecording: false,
  mediaRecorder: null,
  audioChunks: [],
  recognition: null,
  useWhisperAPI: false,
};

// ==================== DOM ELEMENTS ====================
const questionInput = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const sendText = document.getElementById("send-text");
const sendSpinner = document.getElementById("send-spinner");
const messagesDiv = document.getElementById("messages");
const viewToggle = document.getElementById("view-toggle");
const contactToggle = document.getElementById("contact-toggle");
const careerStageSelect = document.getElementById("career-stage");
const industrySelect = document.getElementById("industry");
const topKInput = document.getElementById("top-k");
const statsBar = document.getElementById("stats-bar");
const statTotal = document.getElementById("stat-total");
const statMode = document.getElementById("stat-mode");
const statAvg = document.getElementById("stat-avg");
const insightTable = document.getElementById("insight-table");
const insightRows = document.getElementById("insight-rows");
const closeInsightBtn = document.getElementById("close-insight");
const resumeFileInput = document.getElementById("resume-file-input");
const uploadResumeBtn = document.getElementById("upload-resume-btn");
const uploadStatus = document.getElementById("upload-status");
let currentQuery = '';
let modalRequestInProgress = false;

// Voice elements
const micButton = document.getElementById("mic-button");
const micIcon = document.getElementById("mic-icon");
const voiceStatus = document.getElementById("voice-status");

// ==================== TOGGLE HANDLERS ====================
viewToggle.addEventListener("click", () => {
  state.insightView = !state.insightView;
  viewToggle.classList.toggle("active");
});

contactToggle.addEventListener("click", () => {
  state.showContacts = !state.showContacts;
  contactToggle.classList.toggle("active");
});

closeInsightBtn.addEventListener("click", () => {
  insightTable.classList.remove("visible");
});

// ==================== VOICE UI HELPERS ====================
function showVoiceStatus(message, duration = 0) {
  voiceStatus.textContent = message;
  voiceStatus.classList.add("visible");

  if (duration > 0) {
    setTimeout(() => {
      voiceStatus.classList.remove("visible");
    }, duration);
  }
}

function hideVoiceStatus() {
  voiceStatus.classList.remove("visible");
}

// ==================== ADD MESSAGE ====================
function addMessage(text, role) {
  const bubble = document.createElement("div");
  bubble.className = `message ${role}`;

  const pre = document.createElement("pre");
  pre.textContent = text;
  bubble.appendChild(pre);

  messagesDiv.appendChild(bubble);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// ==================== BUILD FILTERS ====================
function getFilters() {
  const filters = {
    top_k: parseInt(topKInput.value) || 10,
  };

  const careerStage = careerStageSelect.value;
  if (careerStage) filters.career_stage = careerStage;

  const industry = industrySelect.value;
  if (industry) filters.industry = industry;

  return filters;
}

// ==================== UPDATE STATS ====================
function updateStats(results) {
  statsBar.classList.add("visible");
  statTotal.textContent = results.total_found || 0;
  statMode.textContent = results.search_mode || "vector";

  const candidates = results.candidates || [];
  if (candidates.length > 0) {
    const avgMatch =
      candidates.reduce((sum, c) => sum + (c.match?.match_percentage || 0), 0) /
      candidates.length;
    statAvg.textContent = Math.round(avgMatch) + "%";
  } else {
    statAvg.textContent = "0%";
  }
}

// ==================== RENDER INSIGHT TABLE ====================
function renderInsight(results) {
  insightRows.innerHTML = "";
  const candidates = results.candidates || [];

  if (candidates.length === 0) {
    insightRows.innerHTML =
      '<div class="insight-row" style="text-align: center; color: var(--text-muted)">No candidates found</div>';
    insightTable.classList.add("visible");
    return;
  }

  candidates.forEach((cand, idx) => {
    const match = cand.match || {};
    const matchPct = match.match_percentage || 0;

    let badgeClass = "partial";
    if (matchPct >= 80) badgeClass = "";
    else if (matchPct >= 60) badgeClass = "good";

    const row = document.createElement("div");
    row.className = "insight-row";

    const location = cand.location || "Unknown";

    row.innerHTML = `
      <div class="candidate-header">
        <div>
          <div class="candidate-name">${idx + 1}. ${cand.name || "Unknown"}</div>
          <div class="candidate-details">
            <span>📍 ${location}</span>
            <span>💼 ${cand.career_stage || "Unknown"}</span>
            <span>🏢 ${cand.primary_industry || "Unknown"}</span>
            <span>📅 ${cand.total_experience_years || 0} years</span>
          </div>
        </div>
        <div class="match-badge ${badgeClass}">${matchPct}% Match</div>
      </div>

      ${
        cand.summary
          ? `
          <div style="margin-bottom: 0.75rem; color: var(--text-muted); font-size: 0.9rem">
            <div class="summary-text" id="summary-${idx}">
              ${cand.summary.length > 200 ? cand.summary.substring(0, 200) + '...' : cand.summary}
            </div>
            ${cand.summary.length > 200 ? `
              <button 
                class="expand-summary-btn" 
                data-index="${idx}"
                data-full-text="${escapeHtml(cand.summary)}"
                style="color: var(--accent); background: none; border: none; cursor: pointer; font-size: 0.85rem; text-decoration: underline; padding: 0.25rem 0; margin-top: 0.25rem;"
              >
                Read more
              </button>
            ` : ''}
          </div>
          `
          : ""
      }

      <div>
        <strong style="font-size: 0.9rem">Skills:</strong>
        <div class="skill-list">
          ${(match.matched_skills || [])
            .map((s) => `<span class="skill-tag matched">✓ ${s}</span>`)
            .join("")}
          ${(match.missing_skills || [])
            .map((s) => `<span class="skill-tag missing">✗ ${s}</span>`)
            .join("")}
        </div>
      </div>

      ${
        state.showContacts && (cand.email || cand.phone || cand.linkedin_url)
          ? `
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.875rem; color: var(--text-muted)">
          ${cand.email ? `<div>📧 ${cand.email}</div>` : ""}
          ${cand.phone ? `<div>📞 ${cand.phone}</div>` : ""}
          ${
            cand.linkedin_url
              ? `<div>🔗 <a href="${cand.linkedin_url}" target="_blank" style="color: var(--accent)">${cand.linkedin_url}</a></div>`
              : ""
          }
        </div>
      `
          : ""
      }
    `;

    insightRows.appendChild(row);
  });
  document.querySelectorAll(".expand-summary-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const index = btn.dataset.index;
      const fullText = btn.dataset.fullText;
      const summaryDiv = document.getElementById(`summary-${index}`);
      
      if (btn.textContent.trim() === 'Read more') {
        summaryDiv.textContent = fullText;
        btn.textContent = 'Show less';
      } else {
        summaryDiv.textContent = fullText.substring(0, 200) + '...';
        btn.textContent = 'Read more';
      }
    });
  });

  insightTable.classList.add("visible");
}

async function runInsightSearch() {
    const question = document.getElementById('question').value.trim();
    
    if (!question) {
        showError('Please enter a search query');
        return;
    }
    
    // STORE QUERY FOR PHASE 2
    currentQuery = question;
    lastSearchTimestamp = Date.now();
    currentQuery = question;
    // ... rest of your existing search code ...
}

// ==================== PHASE 1: RENDER INSIGHT ROWS (MODIFY) ====================
// Modify your existing renderInsightRows() function:

function renderInsightRows(rows) {
    const container = document.getElementById('insight-rows');
    
    if (!rows || rows.length === 0) {
        container.innerHTML = '<div class="no-results">No candidates found</div>';
        return;
    }
    
    let html = '';
    
    rows.forEach((row, index) => {
        const fitLevelClass = getFitLevelClass(row.fit_level);
        const isCritical = row.has_critical_mismatch;
        
        html += `
            <div class="insight-row ${fitLevelClass}" data-candidate-id="${row.candidate_id}">
                <div class="insight-row-header">
                    <div class="candidate-info">
                        <div class="candidate-name-section">
                            <span class="candidate-name">${escapeHtml(row.candidate)}</span>
                            <span class="fit-badge ${fitLevelClass}">
                                ${row.fit_badge || '—'}
                            </span>
                            ${isCritical ? '<span class="critical-warning">⚠️</span>' : ''}
                        </div>
                        <div class="candidate-position">${escapeHtml(row.position)}</div>
                        ${row.quick_reason ? `<div class="quick-reason">${escapeHtml(row.quick_reason)}</div>` : ''}
                    </div>
                    
                    <div class="insight-actions">
                        <span class="match-chip">${row.match_chip}</span>
                        <button 
                            class="btn-analyze" 
                            onclick="openFitModal('${row.candidate_id}')"
                            title="See detailed fit analysis"
                        >
                            📊 Why?
                        </button>
                    </div>
                </div>
                
                <div class="skill-chips">
                    <div class="matched-skills">
                        ${row.matched && row.matched.length > 0 ? 
                            row.matched.slice(0, 5).map(s => 
                                `<span class="skill-chip matched">✓ ${escapeHtml(s)}</span>`
                            ).join('') : ''
                        }
                    </div>
                    <div class="missing-skills">
                        ${row.missing && row.missing.length > 0 ? 
                            row.missing.slice(0, 3).map(s => 
                                `<span class="skill-chip missing">✗ ${escapeHtml(s)}</span>`
                            ).join('') : ''
                        }
                    </div>
                </div>
                
                ${row.contacts && showContacts ? `
                    <div class="contact-info">
                        ${row.contacts.email ? `<span>📧 ${row.contacts.email}</span>` : ''}
                        ${row.contacts.phone ? `<span>📞 ${row.contacts.phone}</span>` : ''}
                        ${row.contacts.linkedin_url ? `<span>🔗 <a href="${row.contacts.linkedin_url}" target="_blank">LinkedIn</a></span>` : ''}
                    </div>
                ` : ''}
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Helper function to get CSS class for fit level
function getFitLevelClass(fitLevel) {
    const classMap = {
        'excellent': 'fit-excellent',
        'good': 'fit-good',
        'partial': 'fit-partial',
        'poor': 'fit-poor',
        'not_fit': 'fit-not-fit'
    };
    return classMap[fitLevel] || 'fit-unknown';
}

// ==================== PHASE 2: MODAL FUNCTIONS (NEW) ====================

async function openFitModal(candidateId) {
    console.log('Opening fit modal for:', candidateId);
    if (modalRequestInProgress) {
      console.log('Modal request already in progress, ignoring');
      return;
  }
    if (!currentQuery || currentQuery.trim().length === 0) {
     showError('Please perform a search first before analyzing candidates');
     return;
}

// Check query is still relevant
    const queryAge = Date.now() - lastSearchTimestamp;
    if (queryAge > 3600000) {  // 1 hour
      if (!confirm('Search is over 1 hour old. Re-run search for fresh analysis?')) {
          return;
    }
}
    // Show modal with loading state
    showFitModalLoading(candidateId);
    
    try {
        showFitModalLoading(candidateId);
        // Call backend for detailed analysis
        const response = await fetch('/v2/analyze_fit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                job_description: currentQuery,
                candidate_id: candidateId
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }
        
        const analysis = await response.json();
        
        // Render analysis in modal
        erp_count(analysis);
        
    } catch (error) {
        console.error('Fit analysis error:', error);
        showFitModalError(error.message);
    }
}

function showFitModalLoading(candidateId) {
    const modal = createModal();
    
    modal.innerHTML = `
        <div class="modal-content fit-modal">
            <div class="modal-header">
                <h2>Analyzing Candidate Fit...</h2>
                <button class="modal-close" onclick="closeFitModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="loading-spinner">
                    <div class="spinner"></div>
                    <p>Running deep analysis...</p>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Fade in
    setTimeout(() => modal.classList.add('active'), 10);
}

function erp_count(analysis) {
  const modal = document.querySelector('.fit-analysis-modal');
  
  if (!modal) return;
  
  // Check for data quality issues first
  if (analysis.data_quality_issues && analysis.data_quality_issues.length > 0) {
      return renderDataQualityError(analysis);
  }
  
  const fitLevelClass = getFitLevelClass(analysis.fit_level);
  const isCritical = analysis.critical_mismatch !== null;
  
  // Check freshness
  const analyzedAt = new Date(analysis.analyzed_at);
  const ageMinutes = (Date.now() - analyzedAt.getTime()) / 60000;
  
  let freshnessWarning = '';
  if (ageMinutes > 60) {
      freshnessWarning = `
          <div class="freshness-warning">
              ⚠️ This analysis is ${Math.round(ageMinutes / 60)} hours old. 
              Candidate data may have changed. 
              <button onclick="refreshAnalysis('${analysis.candidate_id}')">
                  Refresh Analysis
              </button>
          </div>
      `;
  }
  
  // Check for data warnings
  let warningBanner = '';
  if (analysis.data_warnings && analysis.data_warnings.length > 0) {
      warningBanner = `
          <div class="data-warning-banner">
              <span class="warning-icon">⚠️</span>
              <div>
                  <strong>Data Quality Notice:</strong>
                  <p>${analysis.data_warnings.join(', ')}</p>
                  <p>Analysis may be less accurate. Consider updating candidate profile.</p>
              </div>
          </div>
      `;
  }
  
  modal.innerHTML = `
      <div class="modal-content fit-modal ${fitLevelClass}">
          <!-- Header -->
          <div class="modal-header">
              <div class="modal-title-section">
                  <h2>${escapeHtml(analysis.candidate_name)}</h2>
                  <div class="fit-badge-large ${fitLevelClass}">
                      ${analysis.fit_badge}
                  </div>
                  <div class="fit-score">${analysis.score}% Match</div>
              </div>
              <button class="modal-close" onclick="closeFitModal()">&times;</button>
          </div>
          
          <!-- Body -->
          <div class="modal-body">
              ${freshnessWarning}
              ${warningBanner}
              ${isCritical ? renderCriticalMismatch(analysis) : ''}
              
              <!-- Explanation Section -->
              <div class="analysis-section">
                  <h3>📊 Fit Analysis</h3>
                  <div class="explanation-text">
                      ${formatMarkdown(analysis.explanation)}
                  </div>
              </div>
              
              <!-- Strengths Section -->
              ${analysis.strengths && analysis.strengths.length > 0 ? `
                  <div class="analysis-section strengths-section">
                      <h3>💪 Strengths</h3>
                      <ul class="strength-list">
                          ${analysis.strengths.map(s => 
                              `<li class="strength-item">✓ ${escapeHtml(s)}</li>`
                          ).join('')}
                      </ul>
                  </div>
              ` : ''}
              
              <!-- Weaknesses Section -->
              ${analysis.weaknesses && analysis.weaknesses.length > 0 ? `
                  <div class="analysis-section weaknesses-section">
                      <h3>⚠️ Areas of Concern</h3>
                      <ul class="weakness-list">
                          ${analysis.weaknesses.map(w => 
                              `<li class="weakness-item">✗ ${escapeHtml(w)}</li>`
                          ).join('')}
                      </ul>
                  </div>
              ` : ''}
              
              <!-- Skills Breakdown -->
              <div class="analysis-section skills-breakdown">
                  <h3>🔧 Skills Breakdown</h3>
                  <div class="skills-grid">
                      <div class="skills-column">
                          <h4>Matched Skills (${analysis.matched_skills.length})</h4>
                          <div class="skill-tags">
                              ${analysis.matched_skills.slice(0, 10).map(s => 
                                  `<span class="skill-tag matched">✓ ${escapeHtml(s)}</span>`
                              ).join('')}
                          </div>
                      </div>
                      <div class="skills-column">
                          <h4>Missing Skills (${analysis.missing_skills.length})</h4>
                          <div class="skill-tags">
                              ${analysis.missing_skills.slice(0, 10).map(s => 
                                  `<span class="skill-tag missing">✗ ${escapeHtml(s)}</span>`
                              ).join('')}
                          </div>
                      </div>
                  </div>
              </div>
              
              <!-- Onboarding Estimate -->
              ${analysis.onboarding_estimate ? `
                  <div class="analysis-section onboarding-section">
                      <h3>📅 Onboarding Estimate</h3>
                      <div class="onboarding-grid">
                          <div class="onboarding-item">
                              <span class="label">Time to Productivity:</span>
                              <span class="value">${analysis.onboarding_estimate.time_to_productivity}</span>
                          </div>
                          <div class="onboarding-item">
                              <span class="label">Training Cost:</span>
                              <span class="value">${analysis.onboarding_estimate.training_cost_range}</span>
                          </div>
                          <div class="onboarding-item">
                              <span class="label">Risk Level:</span>
                              <span class="value risk-${analysis.onboarding_estimate.risk_level.toLowerCase()}">
                                  ${analysis.onboarding_estimate.risk_level}
                              </span>
                          </div>
                          ${analysis.onboarding_estimate.notes ? `
                              <div class="onboarding-item full-width">
                                  <span class="label">Notes:</span>
                                  <span class="value">${escapeHtml(analysis.onboarding_estimate.notes)}</span>
                              </div>
                          ` : ''}
                      </div>
                  </div>
              ` : ''}
              
              <!-- Recommendation Section -->
              <div class="analysis-section recommendation-section ${fitLevelClass}">
                  <h3>📋 Hiring Recommendation</h3>
                  <div class="recommendation-text">
                      ${formatMarkdown(analysis.recommendation)}
                  </div>
              </div>
          </div>
          
          <!-- Footer -->
          <div class="modal-footer">
              <button class="btn-secondary" onclick="closeFitModal()">Close</button>
              ${analysis.fit_level === 'excellent' || analysis.fit_level === 'good' ? `
                  <button class="btn-primary" onclick="scheduleInterview('${analysis.candidate_id}')">
                      📅 Schedule Interview
                  </button>
              ` : ''}
          </div>
      </div>
  `;
}
function renderDataQualityError(analysis) {
  const modal = document.querySelector('.fit-analysis-modal');
  
  modal.innerHTML = `
      <div class="modal-content fit-modal">
          <div class="modal-header">
              <h2>⚠️ Incomplete Candidate Data</h2>
              <button class="modal-close" onclick="closeFitModal()">&times;</button>
          </div>
          <div class="modal-body">
              <div class="error-state">
                  <p>Cannot analyze this candidate due to missing information:</p>
                  <ul class="missing-data-list">
                      ${analysis.data_quality_issues.map(issue => 
                          `<li>❌ ${escapeHtml(issue)}</li>`
                      ).join('')}
                  </ul>
                  <p>Please update the candidate record with complete information.</p>
              </div>
          </div>
          <div class="modal-footer">
              <button class="btn-secondary" onclick="closeFitModal()">Close</button>
              <button class="btn-primary" onclick="editCandidate('${analysis.candidate_id}')">
                  Edit Candidate
              </button>
          </div>
      </div>
  `;
}
function renderCriticalMismatch(analysis) {
    const mismatch = analysis.critical_mismatch;
    
    return `
        <div class="critical-mismatch-banner">
            <div class="banner-icon">🚨</div>
            <div class="banner-content">
                <h3>Critical Mismatch Detected</h3>
                <p class="mismatch-type">${mismatch.type.replace(/_/g, ' ').toUpperCase()}</p>
                <div class="mismatch-details">
                    <div class="mismatch-item">
                        <span class="label">Required:</span>
                        <span class="value">${escapeHtml(mismatch.required)}</span>
                    </div>
                    <div class="mismatch-item">
                        <span class="label">Candidate Has:</span>
                        <span class="value">${escapeHtml(mismatch.candidate_has)}</span>
                    </div>
                </div>
                <p class="mismatch-reason">${escapeHtml(mismatch.short_reason)}</p>
            </div>
        </div>
    `;
}

function closeFitModal() {
    const modal = document.querySelector('.fit-analysis-modal');
    if (modal) {
        modal.classList.remove('active');
        setTimeout(() => modal.remove(), 300);
    }
}

function showFitModalError(message) {
    const modal = document.querySelector('.fit-analysis-modal');
    
    if (!modal) return;
    
    modal.innerHTML = `
        <div class="modal-content fit-modal">
            <div class="modal-header">
                <h2>Analysis Failed</h2>
                <button class="modal-close" onclick="closeFitModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="error-message">
                    <span class="error-icon">❌</span>
                    <p>${escapeHtml(message)}</p>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn-secondary" onclick="closeFitModal()">Close</button>
            </div>
        </div>
    `;
}

// Helper: Create modal container
function createModal() {
    // Remove existing modal if any
    const existing = document.querySelector('.fit-analysis-modal');
    if (existing) {
        existing.remove();
    }
    
    const modal = document.createElement('div');
    modal.className = 'fit-analysis-modal';
    return modal;
}

// Helper: Format markdown-like text
function formatMarkdown(text) {
    if (!text) return '';
    
    // Convert **bold** to <strong>
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert line breaks to <br>
    text = text.replace(/\n/g, '<br>');
    
    // Convert bullet points
    text = text.replace(/^[-•]\s/gm, '• ');
    
    return text;
}

// Helper: Escape HTML
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Placeholder for interview scheduling
function scheduleInterview(candidateId) {
    alert(`Interview scheduling for ${candidateId} - integrate with your calendar system`);
}
// ==================== MAIN SEARCH (CHAT / INSIGHT) ====================
async function search() {
  const question = questionInput.value.trim();
  if (!question) return;

  questionInput.value = "";
  addMessage(question, "user");

  sendBtn.disabled = true;
  sendText.style.display = "none";
  sendSpinner.style.display = "inline-block";

  const filters = getFilters();
  const endpoint = state.insightView ? "/v2/insight" : "/v2/chat";

  const payload = {
    question: question,
    filters: filters,
    show_contacts: state.showContacts,
  };

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      addMessage(`❌ Error: ${data.error}`, "bot");
      return;
    }

    if (state.insightView) {
      if (data.rows && data.rows.length > 0) {
        const results = {
          candidates: data.rows.map((row) => ({
            name: row.candidate,
            candidate_id: row.candidate_id,
            career_stage: row.position.split("·")[0]?.trim() || "",
            primary_industry: row.position.split("·")[1]?.trim() || "",
            location_city: "",
            location_state: "",
            location_country: "",
            total_experience_years: 0,
            summary: row.why,
            match: {
              match_percentage: parseInt(
                row.match_chip.match(/\((\d+)%\)/)?.[1] || 0
              ),
              matched_skills: row.matched || [],
              missing_skills: row.missing || [],
            },
            email: row.contacts?.email,
            phone: row.contacts?.phone,
            linkedin_url: row.contacts?.linkedin_url,
          })),
          total_found: data.total_matched || data.rows.length,
          search_mode: "insight",
        };

        updateStats(results);
        renderInsight(results);
        addMessage(
          `📊 Found ${data.rows.length} candidates. View the insight table below.`,
          "bot"
        );
      } else {
        addMessage("No candidates found.", "bot");
      }
    } else {
      addMessage(data.answer || data.text || "No response", "bot");

      if (data.answer && data.answer.includes("Found")) {
        const match = data.answer.match(/Found (\d+) candidates/);
        if (match) {
          statsBar.classList.add("visible");
          statTotal.textContent = match[1];
          statMode.textContent = "chat";
        }
      }
    }
  } catch (err) {
    console.error(err);
    addMessage(`❌ Error: ${err.message}`, "bot");
  } finally {
    sendBtn.disabled = false;
    sendText.style.display = "inline";
    sendSpinner.style.display = "none";
  }
}

// ==================== RESUME UPLOAD (ASYNC + POLLING) ====================
uploadResumeBtn.addEventListener("click", async () => {
  const file = resumeFileInput.files[0];
  if (!file) {
    alert("Please select a file");
    return;
  }

  uploadResumeBtn.disabled = true;
  uploadResumeBtn.textContent = "Uploading...";
  uploadStatus.textContent = "⏳ Uploading...";
  uploadStatus.style.color = "var(--text-muted)";
  console.log('=== UPLOAD STARTED ===');
  console.log('File name:', file.name);
  console.log('Is ZIP?', file.name.endsWith('.zip'));

  try {
    // ====== CHECK IF ZIP FILE ======
    if (file.name.endsWith('.zip')) {
      // Use bulk upload endpoint
      console.log('→ Using BULK endpoint: /v2/bulk_upload_resumes');
      await uploadBulkResumes(file);
    } else {
      // Use single upload endpoint
      console.log('→ Using SINGLE endpoint: /v2/upload_resume_async');

      await uploadSingleResume(file);
    }
    
    resumeFileInput.value = "";
    
  } catch (err) {
    console.error(err);
    uploadStatus.textContent = `❌ ${err.message}`;
    uploadStatus.style.color = "var(--error)";
    uploadResumeBtn.disabled = false;
    uploadResumeBtn.textContent = "Upload";
  }
});

// ====== BULK UPLOAD (for ZIP files) ======
async function uploadBulkResumes(zipFile) {
  const formData = new FormData();
  formData.append('files', zipFile); // Note: 'files' plural
  
  const response = await fetch('/v2/bulk_upload_resumes', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Bulk upload failed");
  }
  
  const data = await response.json();
  const jobId = data.job_id;
  
  uploadStatus.textContent = `✅ Processing ${data.total_files} resumes (Job: ${jobId.substring(0, 8)}...)`;
  uploadStatus.style.color = "var(--success)";
  
  console.log(`Bulk upload started: ${data.total_files} resumes`);
  
  // Poll Celery job status for bulk upload
  pollCeleryJobStatus(jobId, true);  // true = bulk mode
}

// ====== SINGLE UPLOAD (for individual files) ======
async function uploadSingleResume(file) {
  const formData = new FormData();
  formData.append('file', file); // Note: 'file' singular
  
  const response = await fetch('/v2/upload_resume_async', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Upload failed");
  }
  
  const data = await response.json();
  const jobId = data.job_id;
  
  uploadStatus.textContent = `⏳ Processing resume (Job: ${jobId.substring(0, 8)}...)`;
  
  console.log('Single upload started');
  
  // Poll job status (BackgroundTasks)
  pollJobStatus(jobId);
}

// ====== POLL JOB STATUS (for single uploads using BackgroundTasks) ======
async function pollJobStatus(jobId) {
  const maxAttempts = 60;
  let attempts = 0;

  const pollInterval = setInterval(async () => {
    attempts++;

    try {
      const response = await fetch(`/v2/jobs/${jobId}`);

      if (!response.ok) {
        throw new Error("Failed to fetch job status");
      }

      const job = await response.json();

      if (job.status === "completed") {
        clearInterval(pollInterval);

        uploadStatus.textContent = `✅ Resume processed! Candidate: ${job.result.name} (ID: ${job.result.candidate_id})`;
        uploadStatus.style.color = "var(--success)";

        uploadResumeBtn.disabled = false;
        uploadResumeBtn.textContent = "Upload";

        addMessage(
          `Resume uploaded successfully!\nCandidate: ${job.result.name}\nID: ${job.result.candidate_id}\nSkills: ${job.result.skills_count} detected`,
          "bot"
        );
      } else if (job.status === "failed") {
        clearInterval(pollInterval);

        uploadStatus.textContent = `❌ Processing failed: ${job.error}`;
        uploadStatus.style.color = "var(--error)";

        uploadResumeBtn.disabled = false;
        uploadResumeBtn.textContent = "Upload";
      } else if (job.status === "processing") {
        uploadStatus.textContent = `⏳ Processing resume... (${attempts}s)`;
      } else {
        uploadStatus.textContent = `⏳ Queued for processing... (${attempts}s)`;
      }

      if (attempts >= maxAttempts) {
        clearInterval(pollInterval);
        uploadStatus.textContent = "⚠️ Processing taking longer than expected. Check back later.";
        uploadStatus.style.color = "var(--warning)";
        uploadResumeBtn.disabled = false;
        uploadResumeBtn.textContent = "Upload";
      }
    } catch (err) {
      console.error("Poll error:", err);
    }
  }, 1000);
}

// ====== POLL CELERY JOB STATUS (for bulk uploads) ======
async function pollCeleryJobStatus(taskId, isBulk = false) {
  const maxAttempts = isBulk ? 120 : 60;  // 2 mins for single, 4 mins for bulk
  let attempts = 0;

  const pollInterval = setInterval(async () => {
    attempts++;

    try {
      const response = await fetch(`/v2/jobs/celery/${taskId}`);

      if (!response.ok) {
        throw new Error("Failed to fetch job status");
      }

      const job = await response.json();

      if (job.status === "completed") {
        clearInterval(pollInterval);

        if (isBulk) {
          const result = job.result || {};
          uploadStatus.textContent = `✅ Bulk upload complete! Processed ${result.successful || 0} of ${result.total || 0} resumes`;
        } else {
          uploadStatus.textContent = `✅ Resume processed! Candidate: ${job.result?.name || 'Unknown'}`;
        }
        
        uploadStatus.style.color = "var(--success)";
        uploadResumeBtn.disabled = false;
        uploadResumeBtn.textContent = "Upload";

        if (!isBulk && job.result) {
          addMessage(
            `Resume uploaded successfully!\nCandidate: ${job.result.name}\nID: ${job.result.candidate_id}`,
            "bot"
          );
        }
      } else if (job.status === "failed") {
        clearInterval(pollInterval);

        uploadStatus.textContent = `❌ Processing failed: ${job.error || 'Unknown error'}`;
        uploadStatus.style.color = "var(--error)";

        uploadResumeBtn.disabled = false;
        uploadResumeBtn.textContent = "Upload";
      } else {
        const progress = job.progress || 0;
        if (isBulk && progress > 0) {
          uploadStatus.textContent = `⏳ Processing bulk upload... ${progress}%`;
        } else {
          uploadStatus.textContent = `⏳ Processing... (${attempts}s)`;
        }
      }

      if (attempts >= maxAttempts) {
        clearInterval(pollInterval);
        uploadStatus.textContent = "⚠️ Processing taking longer than expected. Check back later.";
        uploadStatus.style.color = "var(--warning)";
        uploadResumeBtn.disabled = false;
        uploadResumeBtn.textContent = "Upload";
      }
    } catch (err) {
      console.error("Poll error:", err);
    }
  }, 2000);  // Poll every 2 seconds
}

// ==================== VOICE INPUT (BROWSER SR + WHISPER FALLBACK) ====================
function initializeSpeechRecognition() {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    console.warn("Speech Recognition not supported; using Whisper fallback");
    state.useWhisperAPI = true;
    return false;
  }

  state.recognition = new SpeechRecognition();
  state.recognition.continuous = false;
  state.recognition.interimResults = true;
  state.recognition.lang = "en-US";
  state.recognition.maxAlternatives = 1;

  state.recognition.onstart = () => {
    showVoiceStatus("🎙️ Listening...");
  };

  state.recognition.onresult = (event) => {
    let interimTranscript = "";
    let finalTranscript = "";

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += transcript;
      } else {
        interimTranscript += transcript;
      }
    }

    if (interimTranscript) {
      questionInput.value = interimTranscript;
      showVoiceStatus(`🎙️ "${interimTranscript}"`);
    }

    if (finalTranscript) {
      questionInput.value = finalTranscript;
      showVoiceStatus("✅ Got it!", 2000);
    }
  };

  state.recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);

    if (event.error === "no-speech") {
      showVoiceStatus("❌ No speech detected", 3000);
    } else if (event.error === "not-allowed" || event.error === "permission-denied") {
      showVoiceStatus("❌ Microphone access denied", 3000);
      alert("Please allow microphone access in your browser settings to use voice input.");
    } else if (event.error === "network") {
      showVoiceStatus("❌ Network error", 3000);
    } else {
      showVoiceStatus("❌ Error: " + event.error, 3000);
    }

    stopRecording();
  };

  state.recognition.onend = () => {
    stopRecording();
  };

  return true;
}

async function startRecording() {
  state.isRecording = true;
  micButton.classList.add("recording");
  micIcon.textContent = "⏹️";

  // Browser SpeechRecognition preferred
  if (state.recognition && !state.useWhisperAPI) {
    try {
      state.recognition.start();
    } catch (e) {
      console.error("Failed to start speech recognition:", e);
      showVoiceStatus("❌ Failed to start", 3000);
      stopRecording();
    }
  } else {
    // Whisper fallback: record audio
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        },
      });

      state.mediaRecorder = new MediaRecorder(stream);
      state.audioChunks = [];

      state.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          state.audioChunks.push(event.data);
        }
      };

      state.mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(state.audioChunks, { type: "audio/webm" });
        await transcribeWithWhisper(audioBlob);

        stream.getTracks().forEach((track) => track.stop());
      };

      state.mediaRecorder.start();
      showVoiceStatus("🎙️ Recording for Whisper API...");
    } catch (e) {
      console.error("Microphone access error:", e);
      showVoiceStatus("❌ Microphone access denied", 3000);
      alert("Please allow microphone access to use voice input.");
      stopRecording();
    }
  }
}
async function openFitModal(candidateId) {
  const maxRetries = 2;
  let attempt = 0;
  
  while (attempt <= maxRetries) {
      try {
          showFitModalLoading(candidateId);
          
          const response = await fetch('/v2/analyze_fit', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                  job_description: currentQuery,
                  candidate_id: candidateId
              }),
              timeout: 30000  // 30 second timeout
          });
          
          if (!response.ok) {
              const error = await response.json();
              
              // Don't retry on 4xx errors (client errors)
              if (response.status >= 400 && response.status < 500) {
                  throw new Error(error.detail || 'Analysis failed');
              }
              
              // Retry on 5xx errors (server errors)
              if (response.status >= 500 && attempt < maxRetries) {
                  attempt++;
                  await sleep(1000 * attempt);  // Exponential backoff
                  continue;
              }
              
              throw new Error(error.detail || 'Server error');
          }
          
          const analysis = await response.json();
          erp_count(analysis);
          return;  // Success, exit
          
      } catch (error) {
          if (attempt >= maxRetries) {
              console.error('Fit analysis failed after retries:', error);
              showFitModalError(error.message);
              return;
          }
          
          attempt++;
          await sleep(1000 * attempt);
      }
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
function stopRecording() {
  state.isRecording = false;
  micButton.classList.remove("recording");
  micIcon.textContent = "🎤";

  if (state.recognition) {
    try {
      state.recognition.stop();
    } catch (e) {
      // already stopped
    }
  }

  if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
    state.mediaRecorder.stop();
  }
}

// Transcribe audio via Whisper API
async function transcribeWithWhisper(audioBlob) {
  micButton.classList.add("processing");
  micIcon.textContent = "⏳";
  showVoiceStatus("⏳ Transcribing with Whisper API...");

  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");

  try {
    const response = await fetch("/v2/transcribe", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Transcription failed");
    }

    const data = await response.json();

    if (data.text) {
      questionInput.value = data.text;
      showVoiceStatus("✅ Transcribed!", 2000);
    } else {
      throw new Error("No transcription returned");
    }
  } catch (error) {
    console.error("Whisper transcription error:", error);
    showVoiceStatus("❌ Transcription failed", 3000);
    addMessage("Transcription failed: " + error.message, "bot");
  } finally {
    micButton.classList.remove("processing");
    micIcon.textContent = "🎤";
  }
}

// ==================== EVENT LISTENERS ====================
// Chat
sendBtn.addEventListener("click", search);
questionInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") search();
});

// Microphone button
micButton.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();

  if (state.isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

// Walkie-talkie mode (SPACE to hold)
let spaceHeld = false;

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target !== questionInput && !spaceHeld) {
    e.preventDefault();
    spaceHeld = true;
    if (!state.isRecording) {
      startRecording();
      showVoiceStatus("🎙️ Hold SPACEBAR to speak...");
    }
  }
});

document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && spaceHeld) {
    e.preventDefault();
    spaceHeld = false;
    if (state.isRecording) {
      stopRecording();
    }
  }
});

// Initialize on page load
window.addEventListener("DOMContentLoaded", () => {
  const speechSupported = initializeSpeechRecognition();

  if (speechSupported) {
    addMessage(
      "🎤 Voice input is ready (Browser Speech Recognition). Click the microphone or hold SPACEBAR to speak.",
      "bot"
    );
  } else {
    addMessage(
      "🎤 Voice input ready (using Whisper API fallback). Click the microphone to speak.",
      "bot"
    );
  }

  questionInput.focus();
});
