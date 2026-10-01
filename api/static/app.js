"use strict";

// Demo client for the check API. Every node is built with textContent:
// API data is never interpreted as HTML.

const WAIT_SECONDS = 25;
const RETRY_DELAY_MS = 3000;

const form = document.getElementById("check-form");
const textArea = document.getElementById("news-text");
const button = document.getElementById("verify");
const statusEl = document.getElementById("status");
const formError = document.getElementById("form-error");
const resultsEl = document.getElementById("results");

let timer = null;
let startedAt = 0;
let lastStatus = "";

function updateButton() {
  button.disabled = textArea.disabled || textArea.value.trim() === "";
}

function setBusy(busy) {
  textArea.disabled = busy;
  statusEl.classList.toggle("busy", busy);
  updateButton();
}

function showFormError(message) {
  formError.textContent = message;
  formError.hidden = !message;
}

function detailMessage(body, fallback) {
  if (body && typeof body.detail === "string") return body.detail;
  if (body && Array.isArray(body.detail) && body.detail.length) {
    return body.detail.map((d) => d.msg || String(d)).join("; ");
  }
  return fallback;
}

function formatElapsed(ms) {
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = String(total % 60).padStart(2, "0");
  return `${m}:${s}`;
}

// Submission-to-finish time from the server timestamps, so a check
// reopened later shows how long it took, not how long ago it was sent.
function finishedDuration(check) {
  const created = Date.parse(check.created_at);
  const finished = Date.parse(check.finished_at);
  const ms =
    Number.isNaN(created) || Number.isNaN(finished)
      ? Date.now() - startedAt
      : finished - created;
  return Math.max(0, ms);
}

function renderStatus() {
  statusEl.textContent = `${lastStatus} · ${formatElapsed(Date.now() - startedAt)}`;
}

function startTimer(fromIso) {
  const parsed = fromIso ? Date.parse(fromIso) : NaN;
  startedAt = Number.isNaN(parsed) ? Date.now() : Math.min(parsed, Date.now());
  stopTimer();
  timer = setInterval(renderStatus, 1000);
}

function stopTimer() {
  if (timer !== null) clearInterval(timer);
  timer = null;
}

function describe(check) {
  if (check.status === "queued") {
    return check.queue_position ? `Queued (#${check.queue_position})` : "Queued";
  }
  if (check.status === "running") return "Running";
  return check.status;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function safeHref(url) {
  return typeof url === "string" && /^https?:\/\//i.test(url) ? url : null;
}

function renderCard(result) {
  const card = el("article", "card");
  const head = el("div", "card-head");
  const verdict = String(result.verdict || "UNCERTAIN");
  const badge = el("span", "badge", verdict);
  if (verdict === "CONFIRMED") badge.classList.add("confirmed");
  else if (verdict === "DISINFORMATION") badge.classList.add("disinformation");
  head.appendChild(badge);
  const score = Number(result.final_score);
  if (Number.isFinite(score)) {
    head.appendChild(el("span", "score", `Score ${Math.round(score * 100)}%`));
  }
  card.appendChild(head);
  card.appendChild(el("p", "claim", result.claim_text));
  if (result.explanation) card.appendChild(el("p", "explanation", result.explanation));

  const citations = Array.isArray(result.citations) ? result.citations : [];
  if (citations.length) {
    card.appendChild(el("p", "sources-title", "Sources"));
    const list = el("ol", "sources");
    for (const c of citations) {
      const item = el("li");
      const label = (c && (c.title || c.url)) || "source";
      const href = safeHref(c && c.url);
      if (href) {
        const link = el("a", null, label);
        link.href = href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        item.appendChild(link);
      } else {
        item.textContent = String(label);
      }
      list.appendChild(item);
    }
    card.appendChild(list);
  }
  return card;
}

function renderFinished(check) {
  resultsEl.replaceChildren();
  if (check.status === "failed") {
    resultsEl.appendChild(
      el("p", "error", `The check failed: ${check.error || "unknown error"}`)
    );
  } else if (!check.results || check.results.length === 0) {
    resultsEl.appendChild(el("p", "message", "No check-worthy claims found in this text."));
  } else {
    for (const result of check.results) resultsEl.appendChild(renderCard(result));
  }
}

async function waitForCheck(id) {
  setBusy(true);
  lastStatus = "Waiting";
  let timerStarted = false;
  let wait = 0; // first request returns at once so the status shows immediately
  for (;;) {
    let response;
    try {
      response = await fetch(`/api/v1/checks/${encodeURIComponent(id)}?wait=${wait}`);
    } catch (err) {
      await sleep(RETRY_DELAY_MS);
      continue;
    }
    if (response.status === 404) {
      finish();
      history.replaceState(null, "", location.pathname);
      showFormError("This check no longer exists.");
      return;
    }
    if (!response.ok) {
      await sleep(RETRY_DELAY_MS);
      continue;
    }
    const check = await response.json();
    wait = WAIT_SECONDS;
    if (!timerStarted) {
      startTimer(check.created_at);
      timerStarted = true;
      if (typeof check.text === "string") textArea.value = check.text;
    }
    lastStatus = describe(check);
    renderStatus();
    if (check.status === "completed" || check.status === "failed") {
      const elapsed = formatElapsed(finishedDuration(check));
      finish();
      statusEl.textContent = `${check.status === "completed" ? "Done" : "Failed"} in ${elapsed}`;
      renderFinished(check);
      return;
    }
  }
}

function finish() {
  stopTimer();
  setBusy(false);
  statusEl.textContent = "";
}

async function submit(event) {
  event.preventDefault();
  const text = textArea.value.trim();
  if (!text) return;
  showFormError("");
  resultsEl.replaceChildren();
  setBusy(true);
  lastStatus = "Submitting";
  statusEl.textContent = "Submitting…";

  let response;
  let body = null;
  try {
    response = await fetch("/api/v1/checks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    body = await response.json().catch(() => null);
  } catch (err) {
    finish();
    showFormError("Could not reach the server. Please try again.");
    return;
  }
  if (response.status !== 202 || !body || !body.id) {
    finish();
    showFormError(detailMessage(body, `Request failed (HTTP ${response.status}).`));
    return;
  }
  location.hash = body.id;
  await waitForCheck(body.id);
}

textArea.addEventListener("input", updateButton);
form.addEventListener("submit", submit);

const resumeId = decodeURIComponent(location.hash.slice(1));
if (resumeId) {
  waitForCheck(resumeId);
} else {
  updateButton();
}
