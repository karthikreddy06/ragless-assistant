/**
 * RAGless Web UI Client
 * Strictly adheres to security guidelines: ALL model/web/stored data rendered
 * exclusively via textContent (zero innerHTML of dynamic content).
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const statusPill = document.getElementById("status-pill");
  const statusText = document.getElementById("status-text");
  const userIdInput = document.getElementById("user-id-input");
  const agentForm = document.getElementById("agent-form");
  const goalInput = document.getElementById("goal-input");
  const runBtn = document.getElementById("run-btn");
  const chips = document.querySelectorAll(".chip");

  const workingCard = document.getElementById("working-card");
  const elapsedCounter = document.getElementById("elapsed-counter");
  const errorCard = document.getElementById("error-card");
  const errorTitle = document.getElementById("error-title");
  const errorMessage = document.getElementById("error-message");

  const emptyStateCard = document.getElementById("empty-state-card");
  const answerCard = document.getElementById("answer-card");
  const answerText = document.getElementById("answer-text");
  const verificationBadge = document.getElementById("verification-badge");
  const refusalBanner = document.getElementById("refusal-banner");
  const refusalBannerText = document.getElementById("refusal-banner-text");
  const unresolvedBox = document.getElementById("unresolved-box");
  const unresolvedList = document.getElementById("unresolved-list");

  const traceCard = document.getElementById("trace-card");
  const traceStepCount = document.getElementById("trace-step-count");
  const traceSteps = document.getElementById("trace-steps");
  const verificationChecks = document.getElementById("verification-checks");

  const evidencePanel = document.getElementById("evidence-panel");
  const evidenceCount = document.getElementById("evidence-count");
  const evidenceMockBadge = document.getElementById("evidence-mock-badge");
  const sourcesList = document.getElementById("sources-list");
  const internalEvidence = document.getElementById("internal-evidence");

  const actionsPanel = document.getElementById("actions-panel");
  const memoryWritesList = document.getElementById("memory-writes-list");
  const tasksChangedList = document.getElementById("tasks-changed-list");

  const progressStatusBadge = document.getElementById("progress-status-badge");
  const progressCounts = document.getElementById("progress-counts");
  const progressPercent = document.getElementById("progress-percent");
  const progressBarFill = document.getElementById("progress-bar-fill");
  const profileRole = document.getElementById("profile-role");
  const profileDepartment = document.getElementById("profile-department");

  const tasksContainer = document.getElementById("tasks-container");
  const tasksCountBadge = document.getElementById("tasks-count-badge");

  const statLatency = document.getElementById("stat-latency");
  const statLlmCalls = document.getElementById("stat-llm-calls");
  const statWebCalls = document.getElementById("stat-web-calls");
  const statModel = document.getElementById("stat-model");

  let timerInterval = null;
  let startTime = 0;

  // Initialize UI
  checkHealth();
  refreshUserState();

  // Chips suggestion clicks
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      if (prompt) {
        goalInput.value = prompt;
        goalInput.focus();
      }
    });
  });

  // User ID input changes
  userIdInput.addEventListener("change", () => {
    refreshUserState();
  });

  // Form submission
  agentForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = goalInput.value.trim();
    const userId = userIdInput.value.trim() || "demo-user";

    if (!question) return;

    // Reset error & results
    hideError();
    hideResults();
    if (emptyStateCard) emptyStateCard.classList.add("hidden");
    startWorkingTimer();

    runBtn.disabled = true;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, 150000); // 150s timeout

    try {
      const resp = await fetch("/agent/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, user_id: userId }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      stopWorkingTimer();
      runBtn.disabled = false;

      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({}));
        handleErrorResponse(resp.status, errorData);
        return;
      }

      const data = await resp.json();
      renderAgentResponse(data);
      refreshUserState();
    } catch (err) {
      clearTimeout(timeoutId);
      stopWorkingTimer();
      runBtn.disabled = false;

      if (err.name === "AbortError") {
        showError("Request Timeout (150s)", "The agent execution exceeded the 150s deadline. Please verify local model status.");
      } else {
        showError("Network Error", "Unable to connect to server. Please verify FastAPI backend is running.");
      }
    }
  });

  // Health check
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        const provider = data.provider || "ollama";
        const model = data.model || "qwen2.5:3b";
        statusText.textContent = `${provider} · ${model}`;
        statusPill.className = "status-pill status-ready";
      } else {
        statusText.textContent = "Service degraded";
        statusPill.className = "status-pill status-error";
      }
    } catch {
      statusText.textContent = "Offline";
      statusPill.className = "status-pill status-error";
    }
  }

  // Refresh user state (memory + tasks)
  async function refreshUserState() {
    const userId = userIdInput.value.trim() || "demo-user";
    await Promise.all([fetchMemory(userId), fetchTasks(userId)]);
  }

  async function fetchMemory(userId) {
    try {
      const res = await fetch(`/memory/${encodeURIComponent(userId)}`);
      if (!res.ok) return;
      const data = await res.json();
      const profile = data.profile || {};
      const progress = data.progress || {};

      profileRole.textContent = profile.role || "Not set";
      profileDepartment.textContent = profile.department || "Not set";

      const total = progress.total_items || 0;
      const completed = progress.completed_count || 0;
      const percent = progress.progress_percentage || 0;

      progressCounts.textContent = `${completed} / ${total} Completed`;
      progressPercent.textContent = `${percent}%`;
      progressBarFill.style.width = `${percent}%`;

      const status = profile.onboarding_status || "not_started";
      progressStatusBadge.textContent = status.replace("_", " ");
      progressStatusBadge.className = status === "completed" ? "badge badge-verified" : "badge badge-neutral";
    } catch {
      // Ignore background refresh errors
    }
  }

  async function fetchTasks(userId) {
    try {
      const res = await fetch(`/tasks/${encodeURIComponent(userId)}`);
      if (!res.ok) return;
      const data = await res.json();
      const tasks = data.tasks || [];

      tasksCountBadge.textContent = String(tasks.length);

      // Clear tasks container securely
      while (tasksContainer.firstChild) {
        tasksContainer.removeChild(tasksContainer.firstChild);
      }

      if (tasks.length === 0) {
        const emptyNotice = document.createElement("p");
        emptyNotice.className = "empty-notice";
        emptyNotice.textContent = "No tasks recorded for user.";
        tasksContainer.appendChild(emptyNotice);
        return;
      }

      tasks.forEach((t) => {
        const card = document.createElement("div");
        card.className = `task-card ${t.status === "done" ? "task-done" : "task-open"}`;

        const titleSpan = document.createElement("span");
        titleSpan.className = "task-card-title";
        titleSpan.textContent = `#${t.id} ${t.title}`;

        const statusSpan = document.createElement("span");
        statusSpan.className = `badge ${t.status === "done" ? "badge-verified" : "badge-neutral"}`;
        statusSpan.textContent = t.status;

        card.appendChild(titleSpan);
        card.appendChild(statusSpan);
        tasksContainer.appendChild(card);
      });
    } catch {
      // Ignore background refresh errors
    }
  }

  // Timer helpers
  function startWorkingTimer() {
    workingCard.classList.remove("hidden");
    startTime = Date.now();
    elapsedCounter.textContent = "0.0s";
    timerInterval = setInterval(() => {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      elapsedCounter.textContent = `${elapsed}s`;
    }, 100);
  }

  function stopWorkingTimer() {
    clearInterval(timerInterval);
    workingCard.classList.add("hidden");
  }

  function showError(title, msg) {
    if (emptyStateCard) emptyStateCard.classList.add("hidden");
    errorTitle.textContent = title;
    errorMessage.textContent = msg;
    errorCard.classList.remove("hidden");
  }

  function hideError() {
    errorCard.classList.add("hidden");
  }

  function hideResults() {
    answerCard.classList.add("hidden");
    traceCard.classList.add("hidden");
    evidencePanel.classList.add("hidden");
    actionsPanel.classList.add("hidden");
    if (refusalBanner) refusalBanner.classList.add("hidden");
    answerCard.classList.remove("card-refusal");
  }

  function handleErrorResponse(status, errorData) {
    if (status === 422) {
      showError("Validation Error (422)", errorData.detail || "Input question or parameters failed schema validation.");
    } else if (status === 503) {
      showError("LLM Service Unavailable (503)", errorData.detail || "Local LLM service is offline or unreachable.");
    } else {
      showError(`Request Error (${status})`, errorData.detail || "An unexpected error occurred while processing the request.");
    }
  }

  // Render agent response safely
  function renderAgentResponse(data) {
    if (emptyStateCard) emptyStateCard.classList.add("hidden");

    // 1. Answer Card & Grounded Refusal
    answerText.textContent = data.answer || "No response generated.";
    answerCard.classList.remove("hidden");

    const isVerified = Boolean(data.verification && data.verification.passed);
    verificationBadge.textContent = isVerified ? "Verified" : "Needs Review";
    verificationBadge.className = `badge ${isVerified ? "badge-verified" : "badge-review"}`;

    // Refusal handling
    const unresolved = (data.verification && data.verification.unresolved) || [];
    const answerLower = (data.answer || "").toLowerCase();
    const isRefusal =
      unresolved.some((u) => u.toLowerCase().includes("no evidence found")) ||
      answerLower.includes("could not find") ||
      answerLower.includes("no evidence found") ||
      answerLower.includes("not found in company records") ||
      answerLower.includes("cannot find");

    if (refusalBanner && refusalBannerText) {
      if (isRefusal) {
        refusalBannerText.textContent = "Grounded Refusal: Information not present in verified database records.";
        refusalBanner.classList.remove("hidden");
        answerCard.classList.add("card-refusal");
      } else {
        refusalBanner.classList.add("hidden");
        answerCard.classList.remove("card-refusal");
      }
    }

    // Unresolved list
    while (unresolvedList.firstChild) {
      unresolvedList.removeChild(unresolvedList.firstChild);
    }
    if (unresolved.length > 0) {
      unresolved.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        unresolvedList.appendChild(li);
      });
      unresolvedBox.classList.remove("hidden");
    } else {
      unresolvedBox.classList.add("hidden");
    }

    // 2. Activity Trace
    const planSteps = (data.plan && data.plan.steps) || [];
    const toolResults = data.tool_results || [];
    traceStepCount.textContent = `${planSteps.length} step${planSteps.length === 1 ? "" : "s"}`;

    while (traceSteps.firstChild) {
      traceSteps.removeChild(traceSteps.firstChild);
    }

    planSteps.forEach((step, idx) => {
      const tr = toolResults[idx] || {};
      const stepDiv = document.createElement("div");
      stepDiv.className = "timeline-step";

      const stepNum = document.createElement("div");
      stepNum.className = "step-num";
      stepNum.textContent = String(idx + 1);

      const stepBody = document.createElement("div");
      stepBody.className = "step-body";

      const stepHeader = document.createElement("div");
      stepHeader.className = "step-header";

      const toolSpan = document.createElement("span");
      toolSpan.className = "step-tool";
      toolSpan.textContent = step.tool;

      const sourceSpan = document.createElement("span");
      sourceSpan.className = "step-source";
      sourceSpan.textContent = tr.source || "sqlite";

      const statusBadge = document.createElement("span");
      const success = Boolean(tr.success);
      statusBadge.className = `badge ${success ? "badge-verified" : "badge-failed"}`;
      statusBadge.textContent = success ? "Success" : "Failed";

      stepHeader.appendChild(toolSpan);
      stepHeader.appendChild(sourceSpan);
      stepHeader.appendChild(statusBadge);

      if (tr.source === "mock") {
        const mockBadge = document.createElement("span");
        mockBadge.className = "badge badge-mock";
        mockBadge.textContent = "MOCK DATA";
        stepHeader.appendChild(mockBadge);
      }

      const stepDesc = document.createElement("p");
      stepDesc.className = "step-desc";
      const descText = step.operation
        ? `Operation: ${step.operation} ${step.args ? JSON.stringify(step.args) : ""}`
        : step.input || (step.args ? JSON.stringify(step.args) : "");
      stepDesc.textContent = descText;

      stepBody.appendChild(stepHeader);
      stepBody.appendChild(stepDesc);

      stepDiv.appendChild(stepNum);
      stepDiv.appendChild(stepBody);
      traceSteps.appendChild(stepDiv);
    });

    // Verification Checks
    const checks = (data.verification && data.verification.checks) || [];
    while (verificationChecks.firstChild) {
      verificationChecks.removeChild(verificationChecks.firstChild);
    }
    checks.forEach((chk) => {
      const chkDiv = document.createElement("div");
      chkDiv.className = `check-item ${chk.passed ? "passed" : "failed"}`;

      const iconSpan = document.createElement("span");
      iconSpan.className = "check-icon";
      iconSpan.textContent = chk.passed ? "✓" : "✗";

      const textDiv = document.createElement("div");
      const nameSpan = document.createElement("span");
      nameSpan.className = "check-name";
      nameSpan.textContent = chk.name;

      const detailSpan = document.createElement("span");
      detailSpan.className = "check-detail";
      detailSpan.textContent = chk.detail;

      textDiv.appendChild(nameSpan);
      textDiv.appendChild(detailSpan);

      chkDiv.appendChild(iconSpan);
      chkDiv.appendChild(textDiv);
      verificationChecks.appendChild(chkDiv);
    });

    traceCard.classList.remove("hidden");

    // 3. Evidence Panel
    const sources = data.sources || [];
    let internalItems = [];
    toolResults.forEach((tr) => {
      if (tr.tool !== "web_research" && tr.data) {
        if (Array.isArray(tr.data)) {
          internalItems.push(...tr.data);
        } else if (typeof tr.data === "object") {
          internalItems.push(tr.data);
        }
      }
    });

    const totalEvidenceCount = sources.length + internalItems.length;
    evidenceCount.textContent = `${totalEvidenceCount} items`;

    const hasMockSource = toolResults.some((tr) => tr.source === "mock");
    if (evidenceMockBadge) {
      if (hasMockSource) {
        evidenceMockBadge.classList.remove("hidden");
      } else {
        evidenceMockBadge.classList.add("hidden");
      }
    }

    // Clear sources list
    while (sourcesList.firstChild) {
      sourcesList.removeChild(sourcesList.firstChild);
    }
    if (sources.length > 0) {
      sources.forEach((s) => {
        const itemDiv = document.createElement("div");
        itemDiv.className = "source-item";

        const link = document.createElement("a");
        link.className = "source-link";
        link.href = s.url;
        link.textContent = s.title || s.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";

        itemDiv.appendChild(link);

        // Find matching snippet if available
        toolResults.forEach((tr) => {
          if (tr.tool === "web_research" && Array.isArray(tr.data)) {
            tr.data.forEach((r) => {
              if (r.url === s.url && r.snippet) {
                const quote = document.createElement("div");
                quote.className = "snippet-quote";
                quote.textContent = `“${r.snippet}”`;
                itemDiv.appendChild(quote);
              }
            });
          }
        });

        sourcesList.appendChild(itemDiv);
      });
    } else {
      const noneP = document.createElement("p");
      noneP.className = "empty-notice";
      noneP.textContent = "No external web sources used.";
      sourcesList.appendChild(noneP);
    }

    // Clear internal evidence table
    while (internalEvidence.firstChild) {
      internalEvidence.removeChild(internalEvidence.firstChild);
    }
    if (internalItems.length > 0) {
      const table = document.createElement("table");
      table.className = "data-table";

      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      const thKey = document.createElement("th");
      thKey.textContent = "Record / Field";
      const thVal = document.createElement("th");
      thVal.textContent = "Value";
      headerRow.appendChild(thKey);
      headerRow.appendChild(thVal);
      thead.appendChild(headerRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      internalItems.slice(0, 15).forEach((item) => {
        const row = document.createElement("tr");
        const tdKey = document.createElement("td");
        const tdVal = document.createElement("td");

        if (item.name && item.employees !== undefined) {
          tdKey.textContent = `Department: ${item.name}`;
          tdVal.textContent = `${item.employees} employees`;
        } else if (item.leave_type) {
          tdKey.textContent = `Leave: ${item.leave_type}`;
          tdVal.textContent = `${item.days} days`;
        } else if (item.category && item.key) {
          tdKey.textContent = `${item.category}.${item.key}`;
          tdVal.textContent = String(item.value);
        } else if (item.title && item.day_due !== undefined) {
          tdKey.textContent = `Onboarding: ${item.title}`;
          tdVal.textContent = `Due Day ${item.day_due}`;
        } else {
          tdKey.textContent = "Data item";
          tdVal.textContent = JSON.stringify(item);
        }

        row.appendChild(tdKey);
        row.appendChild(tdVal);
        tbody.appendChild(row);
      });
      table.appendChild(tbody);
      internalEvidence.appendChild(table);
    } else {
      const noneInternal = document.createElement("p");
      noneInternal.className = "empty-notice";
      noneInternal.textContent = "No internal company records returned.";
      internalEvidence.appendChild(noneInternal);
    }

    evidencePanel.classList.remove("hidden");

    // 4. Actions Panel
    const memWrites = data.memory_writes || [];
    const taskWrites = data.tasks_changed || [];
    const hasActions = memWrites.length > 0 || taskWrites.length > 0;

    while (memoryWritesList.firstChild) {
      memoryWritesList.removeChild(memoryWritesList.firstChild);
    }
    while (tasksChangedList.firstChild) {
      tasksChangedList.removeChild(tasksChangedList.firstChild);
    }

    if (hasActions) {
      memWrites.forEach((mw) => {
        const pill = document.createElement("div");
        pill.className = "action-pill";
        const label = document.createElement("span");
        label.textContent = `${mw.operation}: ${mw.item_or_field}`;
        const badge = document.createElement("span");
        badge.className = `badge ${mw.confirmed ? "badge-verified" : "badge-failed"}`;
        badge.textContent = mw.confirmed ? "Confirmed" : "Unconfirmed";
        pill.appendChild(label);
        pill.appendChild(badge);
        memoryWritesList.appendChild(pill);
      });

      taskWrites.forEach((tw) => {
        const pill = document.createElement("div");
        pill.className = "action-pill";
        const label = document.createElement("span");
        label.textContent = `${tw.operation}: #${tw.task_id} "${tw.title}"`;
        const badge = document.createElement("span");
        badge.className = `badge ${tw.confirmed ? "badge-verified" : "badge-failed"}`;
        badge.textContent = tw.confirmed ? "Confirmed" : "Unconfirmed";
        pill.appendChild(label);
        pill.appendChild(badge);
        tasksChangedList.appendChild(pill);
      });

      if (memWrites.length === 0) {
        const none = document.createElement("p");
        none.className = "empty-notice";
        none.textContent = "No memory writes executed.";
        memoryWritesList.appendChild(none);
      }
      if (taskWrites.length === 0) {
        const none = document.createElement("p");
        none.className = "empty-notice";
        none.textContent = "No task writes executed.";
        tasksChangedList.appendChild(none);
      }

      actionsPanel.classList.remove("hidden");
    } else {
      actionsPanel.classList.add("hidden");
    }

    // 5. Footer Stats
    statLatency.textContent = `${data.processing_time_ms || 0} ms`;
    statLlmCalls.textContent = `${data.llm_calls_count || 0} calls`;
    statWebCalls.textContent = `${data.web_calls_count || 0} calls`;
    statModel.textContent = `${data.llm_provider || "ollama"} · ${data.llm_model || "qwen2.5:3b"}`;
  }
});
