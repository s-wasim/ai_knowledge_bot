// =============================================================================
// VeloRelAI Knowledge Bot — API Adapter
// -----------------------------------------------------------------------------
// Single point of contact for all backend data. Every UI component calls into
// this module only — never fetch() directly from a component.
//
// LIVE MODE: calls the FastAPI backend served from the same origin as this
// file. Exported function names, parameters, and callback shapes are
// unchanged from the original mock so no caller-side code needs to change.
// =============================================================================

export const API_BASE = window.location.origin;

// -----------------------------------------------------------------------------
// SSE helper — POST a JSON body, parse a `text/event-stream` response as it
// arrives, and dispatch parsed frames to `on<EventName>` handlers. Returns a
// cancel function that aborts the in-flight request.
// -----------------------------------------------------------------------------
function sseFetch(url, body, handlers) {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => "");
        handlers.onError({ message: text || `HTTP ${res.status}` });
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);

          const eventMatch = /^event:\s*(.+)$/m.exec(frame);
          const dataMatch = /^data:\s*(.+)$/m.exec(frame);
          if (!eventMatch || !dataMatch) continue;

          const eventName = eventMatch[1].trim();
          let data;
          try {
            data = JSON.parse(dataMatch[1]);
          } catch {
            continue;
          }

          const handlerName = "on" + eventName[0].toUpperCase() + eventName.slice(1);
          const handler = handlers[handlerName];
          if (handler) handler(data);
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") {
        handlers.onError({ message: e.message || String(e) });
      }
    }
  })();

  return () => controller.abort();
}

// =============================================================================
// GET /health
// =============================================================================
export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

// =============================================================================
// GET /repos
// =============================================================================
export async function listRepos() {
  const res = await fetch(`${API_BASE}/repos`);
  return res.json();
}

// =============================================================================
// POST /ingest/local  (Server-Sent Events)
// =============================================================================
export function ingestLocal({ path, name }, onProgress, onDone, onError, onWarning) {
  return sseFetch(
    `${API_BASE}/ingest/local`,
    { path, name },
    { onProgress, onDone, onError, onWarning }
  );
}

// =============================================================================
// POST /ingest/github  (Server-Sent Events)
// =============================================================================
export function ingestGithub({ url, branch }, onProgress, onDone, onError, onWarning) {
  return sseFetch(
    `${API_BASE}/ingest/github`,
    { url, branch },
    { onProgress, onDone, onError, onWarning }
  );
}

// =============================================================================
// POST /chat  (Server-Sent Events)
// =============================================================================
export function chat({ repo_id, question, history }, { onNode, onToken, onFinal, onError }) {
  const trimmedHistory = (history || []).slice(-6);
  return sseFetch(
    `${API_BASE}/chat`,
    { repo_id, question, history: trimmedHistory },
    { onNode, onToken, onFinal, onError }
  );
}

// =============================================================================
// GET /repos/{repo_id}/browse?keyword=
// =============================================================================
export async function browse(repo_id, keyword) {
  const res = await fetch(
    `${API_BASE}/repos/${repo_id}/browse?keyword=${encodeURIComponent(keyword || "")}`
  );
  return res.json();
}

// =============================================================================
// GET /repos/{repo_id}/files?path=
// =============================================================================
export async function browseFile(repo_id, path) {
  const res = await fetch(
    `${API_BASE}/repos/${repo_id}/files?path=${encodeURIComponent(path)}`
  );
  return res.json();
}

// =============================================================================
// GET /config/allowlist
// =============================================================================
export async function getAllowlist() {
  const res = await fetch(`${API_BASE}/config/allowlist`);
  return res.json();
}
