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

    const location =
      [cand.location_city, cand.location_state, cand.location_country]
        .filter(Boolean)
        .join(", ") || "Unknown";

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
          ? `<div style="margin-bottom: 0.75rem; color: var(--text-muted); font-size: 0.9rem">${cand.summary}</div>`
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

  insightTable.classList.add("visible");
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
