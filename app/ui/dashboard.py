import json
import threading
from typing import Any, Dict


class SharedUIState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pred_text = "Idle"
        self._topk_lines = []
        self._transcript = ""
        self._debug = False

        self._stop = False
        self._clear_requested = False
        self._toggle_debug_requested = False

        self._running = False
        self._last_error = ""

    def reset_for_run(self, debug: bool) -> None:
        with self._lock:
            self._pred_text = "Running..."
            self._topk_lines = []
            self._transcript = ""
            self._debug = bool(debug)
            self._stop = False
            self._clear_requested = False
            self._toggle_debug_requested = False
            self._running = True
            self._last_error = ""

    def mark_stopped(self, err: str = "") -> None:
        with self._lock:
            self._running = False
            self._stop = False
            if err:
                self._last_error = err
                self._pred_text = "Stopped (error)"
            else:
                self._pred_text = "Stopped"

    def update(self, pred_text, topk_lines, transcript, debug) -> None:
        with self._lock:
            self._pred_text = pred_text
            self._topk_lines = list(topk_lines)
            self._transcript = transcript
            self._debug = bool(debug)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pred_text": self._pred_text,
                "topk_lines": list(self._topk_lines),
                "transcript": self._transcript,
                "debug": self._debug,
                "running": self._running,
                "last_error": self._last_error,
            }

    def request_stop(self) -> None:
        with self._lock:
            self._stop = True

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop

    def request_clear_transcript(self) -> None:
        with self._lock:
            self._clear_requested = True

    def consume_clear_transcript(self) -> bool:
        with self._lock:
            if self._clear_requested:
                self._clear_requested = False
                return True
            return False

    def request_toggle_debug(self) -> None:
        with self._lock:
            self._toggle_debug_requested = True

    def consume_toggle_debug(self) -> bool:
        with self._lock:
            if self._toggle_debug_requested:
                self._toggle_debug_requested = False
                return True
            return False


class SessionController:
    def __init__(self, base_kwargs: Dict[str, Any], state: SharedUIState):
        self._base_kwargs = dict(base_kwargs)
        self._state = state
        self._lock = threading.Lock()
        self._thread = None

    @staticmethod
    def _coerce(overrides: Dict[str, Any]) -> Dict[str, Any]:
        def as_int(k, d):
            try:
                return int(overrides.get(k, d))
            except Exception:
                return d

        def as_float(k, d):
            try:
                return float(overrides.get(k, d))
            except Exception:
                return d

        def as_str(k, d):
            v = overrides.get(k, d)
            return str(v)

        def as_bool(k, d):
            v = overrides.get(k, d)
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() in ("1", "true", "yes", "on")
            return bool(v)

        return {
            "camera": as_int("camera", -1),
            "backend": as_str("backend", "any"),
            "cam_width": as_int("cam_width", 640),
            "cam_height": as_int("cam_height", 480),
            "clip_len": as_int("clip_len", 32),
            "topk": as_int("topk", 5),
            "threshold": as_float("threshold", 0.07),
            "margin_threshold": as_float("margin_threshold", 0.008),
            "ema": as_float("ema", 0.6),
            "infer_every": as_int("infer_every", 2),
            "vote_window": as_int("vote_window", 5),
            "min_votes": as_int("min_votes", 3),
            "cooldown": as_int("cooldown", 2),
            "stuck_window": as_int("stuck_window", 10),
            "stuck_majority": as_int("stuck_majority", 8),
            "stuck_conf_max": as_float("stuck_conf_max", 0.085),
            "debug_probs": as_bool("debug_probs", False),
            "show_preview": as_bool("show_preview", False),
            "virtual_cam_enabled": as_bool("virtual_cam_enabled", True),
            "virtual_cam_fps": as_int("virtual_cam_fps", 60),
            "virtual_cam_mirror": as_bool("virtual_cam_mirror", False),
            "llm_enabled": as_bool("llm_enabled", True),
            "llm_model": as_str("llm_model", "qwen3:4b"),
            "llm_interval_sec": as_float("llm_interval_sec", 1.5),
            "llm_min_tokens": as_int("llm_min_tokens", 2),
        }

    # FIX 1: Import run_session at module level (top of _runner), not nested
    # inside the closure — some pywebview builds shadow the sys.path inside the
    # thread so a deferred import inside a lambda silently fails.
    def start(self, overrides: Dict[str, Any]):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"ok": False, "message": "Session is already running"}

            opts = self._coerce(overrides or {})
            kwargs = dict(self._base_kwargs)
            kwargs.setdefault("show_preview", False)
            kwargs.update(opts)

            self._state.reset_for_run(debug=kwargs.get("debug_probs", False))

            # FIX 2: Capture kwargs by value so mutation after start() can't
            # affect the running thread.
            captured_kwargs = dict(kwargs)
            state_ref = self._state

            def _runner():
                # Import at the top of the thread entry point, not inside a
                # nested closure — avoids silent ImportError in some pywebview
                # environments where sys.path isn't inherited by the thread.
                try:
                    from app.runtime.session import run_session
                except ImportError as exc:
                    state_ref.mark_stopped(err=f"Import error: {exc}")
                    return

                err = ""
                try:
                    run_session(shared_state=state_ref, **captured_kwargs)
                except Exception as exc:
                    err = str(exc)
                finally:
                    state_ref.mark_stopped(err=err)

            self._thread = threading.Thread(target=_runner, daemon=True)
            self._thread.start()

            return {"ok": True, "message": "Session started"}

    def stop(self):
        self._state.request_stop()
        return {"ok": True, "message": "Stop requested"}

    def defaults(self):
        keys = [
            "camera",
            "backend",
            "cam_width",
            "cam_height",
            "clip_len",
            "topk",
            "threshold",
            "margin_threshold",
            "ema",
            "infer_every",
            "vote_window",
            "min_votes",
            "cooldown",
            "stuck_window",
            "stuck_majority",
            "stuck_conf_max",
            "debug_probs",
            "show_preview",
            "virtual_cam_enabled",
            "virtual_cam_fps",
            "virtual_cam_mirror",
            "llm_enabled",
            "llm_model",
            "llm_interval_sec",
            "llm_min_tokens",
        ]
        return {k: self._base_kwargs.get(k) for k in keys}


class DashboardAPI:
    def __init__(self, state: SharedUIState, controller: SessionController):
        self.state = state
        self.controller = controller

    def get_state(self):
        return {
            "state": self.state.snapshot(),
            "defaults": self.controller.defaults(),
        }

    def start_session(self, options_json="{}"):
        if isinstance(options_json, str):
            try:
                options = json.loads(options_json)
            except Exception:
                options = {}
        elif isinstance(options_json, dict):
            options = options_json
        else:
            options = {}
        return self.controller.start(options)

    def stop_session(self):
        return self.controller.stop()

    def clear_transcript(self):
        self.state.request_clear_transcript()
        return {"ok": True}

    def toggle_debug(self):
        self.state.request_toggle_debug()
        return {"ok": True}

    # camelCase aliases for pywebview JS bridge compatibility
    def startSession(self, options_json="{}"):
        return self.start_session(options_json)

    def stopSession(self):
        return self.stop_session()

    def clearTranscript(self):
        return self.clear_transcript()

    def toggleDebug(self):
        return self.toggle_debug()

    def getState(self):
        return self.get_state()


def _render_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ASL Recognition</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0c0e10;
      --surface:   #13161a;
      --border:    #1f2428;
      --border-hi: #2d3238;
      --muted:     #4a5260;
      --text:      #d6dce8;
      --text-dim:  #7a8494;
      --green:     #00e5a0;
      --green-dim: #004d37;
      --red:       #ff4f6a;
      --red-dim:   #3d0f18;
      --amber:     #ffb830;
      --amber-dim: #3d2a00;
      --blue:      #4da8ff;
      --mono: 'DM Mono', monospace;
      --sans: 'Syne', sans-serif;
      --r: 10px;
    }

    html, body {
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      font-size: 14px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    /* ── Layout ── */
    .shell {
      display: grid;
      grid-template-rows: 56px 1fr;
      grid-template-columns: 300px 1fr;
      height: 100vh;
      gap: 0;
    }

    /* ── Top bar ── */
    .topbar {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 0 24px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
    }
    .topbar-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 6px var(--green);
      flex-shrink: 0;
      transition: background .3s, box-shadow .3s;
    }
    .topbar-dot.idle  { background: var(--muted); box-shadow: none; }
    .topbar-dot.error { background: var(--red);   box-shadow: 0 0 6px var(--red); }
    .topbar-title { font-weight: 700; font-size: 15px; letter-spacing: .04em; }
    .topbar-status {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text-dim);
      margin-left: 4px;
    }
    .topbar-error {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--red);
      margin-left: auto;
      max-width: 360px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    /* ── Sidebar ── */
    .sidebar {
      border-right: 1px solid var(--border);
      background: var(--surface);
      overflow-y: auto;
      padding: 20px 16px 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .section-label {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }

    /* ── Field groups ── */
    .field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .field { display: flex; flex-direction: column; gap: 5px; }
    .field label {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text-dim);
    }
    .field input,
    .field select {
      background: var(--bg);
      border: 1px solid var(--border-hi);
      border-radius: 6px;
      color: var(--text);
      font-family: var(--mono);
      font-size: 12px;
      padding: 7px 10px;
      width: 100%;
      outline: none;
      transition: border-color .15s;
    }
    .field input:focus,
    .field select:focus { border-color: var(--blue); }
    .field select { cursor: pointer; }

    /* ── Buttons ── */
    .btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .btn {
      border: none;
      border-radius: var(--r);
      padding: 10px 0;
      font-family: var(--sans);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      letter-spacing: .03em;
      transition: opacity .15s, transform .1s;
    }
    .btn:active { transform: scale(.97); }
    .btn:disabled { opacity: .35; cursor: not-allowed; }
    .btn-start  { background: var(--green);    color: #001a12; }
    .btn-stop   { background: var(--red-dim);  color: var(--red);   border: 1px solid var(--red); }
    .btn-clear  { background: var(--border);   color: var(--text-dim); }
    .btn-debug  { background: var(--border);   color: var(--text-dim); }
    .btn-debug.active { background: var(--amber-dim); color: var(--amber); border: 1px solid var(--amber); }

    /* ── Main panel ── */
    .main {
      display: grid;
      grid-template-rows: auto 1fr 1fr;
      gap: 0;
      overflow: hidden;
    }

    /* ── Prediction hero ── */
    .pred-hero {
      padding: 28px 32px 20px;
      border-bottom: 1px solid var(--border);
    }
    .pred-label {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: .14em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .pred-value {
      font-family: var(--sans);
      font-size: 42px;
      font-weight: 700;
      line-height: 1;
      color: var(--text);
      letter-spacing: -.01em;
      min-height: 48px;
      transition: color .2s;
    }
    .pred-value.live { color: var(--green); }

    /* ── Top-5 panel ── */
    .panel {
      padding: 20px 28px;
      border-bottom: 1px solid var(--border);
      overflow-y: auto;
    }
    .topk-list { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
    .topk-item {
      display: grid;
      grid-template-columns: 2ch 1fr auto;
      gap: 10px;
      align-items: center;
    }
    .topk-rank { font-family: var(--mono); font-size: 11px; color: var(--muted); }
    .topk-bar-wrap {
      height: 6px;
      background: var(--border);
      border-radius: 3px;
      overflow: hidden;
    }
    .topk-bar {
      height: 100%;
      background: var(--green);
      border-radius: 3px;
      transition: width .3s ease;
    }
    .topk-item:not(:first-child) .topk-bar { background: var(--muted); }
    .topk-label {
      font-family: var(--mono);
      font-size: 12px;
      color: var(--text);
      white-space: nowrap;
    }
    .topk-empty {
      font-family: var(--mono);
      font-size: 12px;
      color: var(--muted);
      padding: 8px 0;
    }

    /* ── Transcript panel ── */
    .transcript-wrap {
      padding: 20px 28px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }
    .transcript-text {
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.9;
      color: var(--text);
      white-space: pre-wrap;
      word-break: break-word;
      flex: 1;
    }
    .transcript-empty {
      font-family: var(--mono);
      font-size: 12px;
      color: var(--muted);
    }

    /* ── API not ready overlay ── */
    #api-overlay {
      position: fixed;
      inset: 0;
      background: rgba(12,14,16,.85);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 999;
      font-family: var(--mono);
      font-size: 13px;
      color: var(--muted);
      gap: 10px;
    }
    .spinner {
      width: 14px; height: 14px;
      border: 2px solid var(--border-hi);
      border-top-color: var(--blue);
      border-radius: 50%;
      animation: spin .7s linear infinite;
      flex-shrink: 0;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    #api-overlay.hidden { display: none; }
  </style>
</head>
<body>

<div id="api-overlay">
  <div class="spinner"></div>
  Connecting to runtime…
</div>

<div class="shell">

  <!-- Top bar -->
  <header class="topbar">
    <div id="status-dot" class="topbar-dot idle"></div>
    <span class="topbar-title">ASL Recognition</span>
    <span id="status-text" class="topbar-status">idle</span>
    <span id="error-text" class="topbar-error"></span>
  </header>

  <!-- Sidebar -->
  <aside class="sidebar">
    <div>
      <div class="section-label">Camera</div>
      <div class="field-grid">
        <div class="field">
          <label>Index (-1 auto)</label>
          <input id="camera" type="number" value="-1" min="-1"/>
        </div>
        <div class="field">
          <label>Backend</label>
          <select id="backend">
            <option value="any">any</option>
            <option value="auto">auto</option>
            <option value="avfoundation">avfoundation</option>
          </select>
        </div>
        <div class="field">
          <label>Width</label>
          <input id="cam_width" type="number" value="640" min="160"/>
        </div>
        <div class="field">
          <label>Height</label>
          <input id="cam_height" type="number" value="480" min="120"/>
        </div>
      </div>
    </div>

    <div>
      <div class="section-label">Inference</div>
      <div class="field-grid">
        <div class="field">
          <label>Threshold</label>
          <input id="threshold" type="number" step="0.001"/>
        </div>
        <div class="field">
          <label>Margin</label>
          <input id="margin_threshold" type="number" step="0.001"/>
        </div>
        <div class="field">
          <label>EMA</label>
          <input id="ema" type="number" step="0.01"/>
        </div>
        <div class="field">
          <label>Infer every</label>
          <input id="infer_every" type="number" min="1"/>
        </div>
        <div class="field">
          <label>Clip length</label>
          <input id="clip_len" type="number" min="4"/>
        </div>
        <div class="field">
          <label>Top K</label>
          <input id="topk" type="number" min="1"/>
        </div>
      </div>
    </div>

    <div>
      <div class="section-label">Voting</div>
      <div class="field-grid">
        <div class="field">
          <label>Window</label>
          <input id="vote_window" type="number" min="1"/>
        </div>
        <div class="field">
          <label>Min votes</label>
          <input id="min_votes" type="number" min="1"/>
        </div>
        <div class="field">
          <label>Cooldown</label>
          <input id="cooldown" type="number" min="0"/>
        </div>
        <div class="field">
          <label>Debug</label>
          <select id="debug_probs">
            <option value="false">off</option>
            <option value="true">on</option>
          </select>
        </div>
        <div class="field">
          <label>Stuck window</label>
          <input id="stuck_window" type="number" min="1"/>
        </div>
        <div class="field">
          <label>Stuck majority</label>
          <input id="stuck_majority" type="number" min="1"/>
        </div>
        <div class="field">
          <label>Stuck conf max</label>
          <input id="stuck_conf_max" type="number" step="0.001" min="0"/>
        </div>
      </div>
    </div>

    <div>
      <div class="section-label">Output</div>
      <div class="field-grid">
        <div class="field">
          <label>Preview window</label>
          <select id="show_preview">
            <option value="false">off</option>
            <option value="true">on</option>
          </select>
        </div>
        <div class="field">
          <label>Virtual cam</label>
          <select id="virtual_cam_enabled">
            <option value="false">off</option>
            <option value="true">on</option>
          </select>
        </div>
        <div class="field">
          <label>Virtual cam FPS</label>
          <input id="virtual_cam_fps" type="number" min="1"/>
        </div>
        <div class="field">
          <label>Virtual cam mirror</label>
          <select id="virtual_cam_mirror">
            <option value="false">off</option>
            <option value="true">on</option>
          </select>
        </div>
      </div>
    </div>

    <div>
      <div class="section-label">LLM</div>
      <div class="field-grid">
        <div class="field">
          <label>LLM enabled</label>
          <select id="llm_enabled">
            <option value="false">off</option>
            <option value="true">on</option>
          </select>
        </div>
        <div class="field">
          <label>Model</label>
          <input id="llm_model" type="text" value="qwen3:4b"/>
        </div>
        <div class="field">
          <label>Rewrite interval</label>
          <input id="llm_interval_sec" type="number" step="0.1" min="0.1"/>
        </div>
        <div class="field">
          <label>Min tokens</label>
          <input id="llm_min_tokens" type="number" min="1"/>
        </div>
      </div>
    </div>

    <div>
      <div class="section-label">Controls</div>
      <div class="btn-row">
        <button id="btn-start" class="btn btn-start" onclick="startSession()">Start</button>
        <button id="btn-stop"  class="btn btn-stop"  onclick="stopSession()">Stop</button>
      </div>
      <div class="btn-row" style="margin-top:8px">
        <button id="btn-clear" class="btn btn-clear" onclick="clearTranscript()">Clear</button>
        <button id="btn-debug" class="btn btn-debug" onclick="toggleDebug()">Debug</button>
      </div>
      <div id="msg" style="font-family:var(--mono);font-size:11px;color:var(--text-dim);margin-top:10px;min-height:16px;"></div>
    </div>
  </aside>

  <!-- Main -->
  <main class="main">
    <!-- Prediction hero -->
    <div class="pred-hero">
      <div class="pred-label">Top prediction</div>
      <div id="pred-value" class="pred-value">—</div>
    </div>

    <!-- Top-5 -->
    <div class="panel">
      <div class="section-label">Top 5</div>
      <div id="topk-list" class="topk-list">
        <div class="topk-empty">No data yet</div>
      </div>
    </div>

    <!-- Transcript -->
    <div class="transcript-wrap">
      <div class="section-label">Transcript</div>
      <div id="transcript" class="transcript-text transcript-empty">Waiting for session…</div>
    </div>
  </main>

</div>

<script>
  let initialized = false;
  let apiReady = false;
  let debugActive = false;

  // ── API readiness ──────────────────────────────────────────────
  function checkApiReady() {
    // pywebview exposes the api object asynchronously after the window loads.
    // Poll until it's available before hiding the overlay.
    if (window.pywebview && window.pywebview.api) {
      apiReady = true;
      document.getElementById('api-overlay').classList.add('hidden');
      refresh();
    } else {
      setTimeout(checkApiReady, 150);
    }
  }
  checkApiReady();

  // ── API call helper ────────────────────────────────────────────
  async function callApi(primary, fallback, ...args) {
    if (!apiReady) return { ok: false, message: 'API not ready' };
    const api = window.pywebview.api;
    try {
      if (typeof api[primary] === 'function') return await api[primary](...args);
      if (fallback && typeof api[fallback] === 'function') return await api[fallback](...args);
      return { ok: false, message: 'Missing method: ' + primary };
    } catch (e) {
      return { ok: false, message: String(e) };
    }
  }

  // ── Read form values ───────────────────────────────────────────
  function readOptions() {
    return {
      camera:           Number(document.getElementById('camera').value),
      backend:          document.getElementById('backend').value,
      cam_width:        Number(document.getElementById('cam_width').value),
      cam_height:       Number(document.getElementById('cam_height').value),
      clip_len:         Number(document.getElementById('clip_len').value),
      topk:             Number(document.getElementById('topk').value),
      threshold:        Number(document.getElementById('threshold').value),
      margin_threshold: Number(document.getElementById('margin_threshold').value),
      ema:              Number(document.getElementById('ema').value),
      infer_every:      Number(document.getElementById('infer_every').value),
      vote_window:      Number(document.getElementById('vote_window').value),
      min_votes:        Number(document.getElementById('min_votes').value),
      cooldown:         Number(document.getElementById('cooldown').value),
      stuck_window:     Number(document.getElementById('stuck_window').value),
      stuck_majority:   Number(document.getElementById('stuck_majority').value),
      stuck_conf_max:   Number(document.getElementById('stuck_conf_max').value),
      debug_probs:      document.getElementById('debug_probs').value === 'true',
      show_preview:     document.getElementById('show_preview').value === 'true',
      virtual_cam_enabled: document.getElementById('virtual_cam_enabled').value === 'true',
      virtual_cam_fps:  Number(document.getElementById('virtual_cam_fps').value),
      virtual_cam_mirror: document.getElementById('virtual_cam_mirror').value === 'true',
      llm_enabled:      document.getElementById('llm_enabled').value === 'true',
      llm_model:        document.getElementById('llm_model').value,
      llm_interval_sec: Number(document.getElementById('llm_interval_sec').value),
      llm_min_tokens:   Number(document.getElementById('llm_min_tokens').value),
    };
  }

  function applyDefaults(d) {
    if (!d) return;
    const set = (id, v) => { if (v !== undefined && v !== null) document.getElementById(id).value = v; };
    set('camera', d.camera);
    set('cam_width', d.cam_width);
    set('cam_height', d.cam_height);
    set('clip_len', d.clip_len);
    set('topk', d.topk);
    set('threshold', d.threshold);
    set('margin_threshold', d.margin_threshold);
    set('ema', d.ema);
    set('infer_every', d.infer_every);
    set('vote_window', d.vote_window);
    set('min_votes', d.min_votes);
    set('cooldown', d.cooldown);
    set('stuck_window', d.stuck_window);
    set('stuck_majority', d.stuck_majority);
    set('stuck_conf_max', d.stuck_conf_max);
    if (d.backend !== undefined) document.getElementById('backend').value = d.backend;
    if (d.debug_probs !== undefined) document.getElementById('debug_probs').value = d.debug_probs ? 'true' : 'false';
    if (d.show_preview !== undefined) document.getElementById('show_preview').value = d.show_preview ? 'true' : 'false';
    if (d.virtual_cam_enabled !== undefined) document.getElementById('virtual_cam_enabled').value = d.virtual_cam_enabled ? 'true' : 'false';
    set('virtual_cam_fps', d.virtual_cam_fps);
    if (d.virtual_cam_mirror !== undefined) document.getElementById('virtual_cam_mirror').value = d.virtual_cam_mirror ? 'true' : 'false';
    if (d.llm_enabled !== undefined) document.getElementById('llm_enabled').value = d.llm_enabled ? 'true' : 'false';
    set('llm_model', d.llm_model);
    set('llm_interval_sec', d.llm_interval_sec);
    set('llm_min_tokens', d.llm_min_tokens);
  }

  // ── Render state ───────────────────────────────────────────────
  function renderState(s) {
    const dot  = document.getElementById('status-dot');
    const stxt = document.getElementById('status-text');
    const etxt = document.getElementById('error-text');
    const pred = document.getElementById('pred-value');

    const running = s.running;
    const err     = s.last_error || '';

    dot.className  = 'topbar-dot ' + (err ? 'error' : running ? '' : 'idle');
    stxt.textContent = err ? 'error' : running ? 'running' : 'idle';
    etxt.textContent = err;

    // Prediction
    pred.textContent = s.pred_text || '—';
    pred.className   = 'pred-value' + (running ? ' live' : '');

    // Top-5 bars
    renderTopK(s.topk_lines || []);

    // Transcript
    const tx = document.getElementById('transcript');
    if (s.transcript) {
      tx.textContent = s.transcript;
      tx.classList.remove('transcript-empty');
    } else {
      tx.textContent = running ? '' : 'Waiting for session…';
      tx.classList.add('transcript-empty');
    }

    // Button states
    document.getElementById('btn-start').disabled = running;
    document.getElementById('btn-stop').disabled  = !running;
  }

  function renderTopK(lines) {
    const el = document.getElementById('topk-list');
    if (!lines.length) {
      el.innerHTML = '<div class="topk-empty">No data yet</div>';
      return;
    }
    // Each line expected format: "LABEL  0.832" or "LABEL: 0.832"
    const items = lines.map((line, i) => {
      const parts = line.trim().split(/\s+/);
      const score = parseFloat(parts[parts.length - 1]) || 0;
      const label = parts.slice(0, -1).join(' ') || line;
      const pct   = Math.round(Math.min(score, 1) * 100);
      return `<div class="topk-item">
        <span class="topk-rank">${i + 1}</span>
        <div class="topk-bar-wrap"><div class="topk-bar" style="width:${pct}%"></div></div>
        <span class="topk-label">${label} <span style="color:var(--muted)">${pct}%</span></span>
      </div>`;
    });
    el.innerHTML = items.join('');
  }

  // ── Poll ───────────────────────────────────────────────────────
  async function refresh() {
    const payload = await callApi('get_state', 'getState');
    if (!payload || !payload.state) return;

    if (!initialized) {
      applyDefaults(payload.defaults || {});
      initialized = true;
    }
    renderState(payload.state);
  }

  // ── Actions ────────────────────────────────────────────────────
  async function startSession() {
    const opts = JSON.stringify(readOptions());
    // FIX: pass as a plain string — pywebview serialises JS→Python arguments
    // correctly when the argument is already a JSON string, but some builds
    // double-encode an object literal. Stringify once here, parse once in Python.
    const res = await callApi('start_session', 'startSession', opts);
    document.getElementById('msg').textContent = res?.message || '';
    if (res?.ok) refresh();
  }

  async function stopSession() {
    const res = await callApi('stop_session', 'stopSession');
    document.getElementById('msg').textContent = res?.message || '';
  }

  async function clearTranscript() {
    await callApi('clear_transcript', 'clearTranscript');
    document.getElementById('transcript').textContent = '';
  }

  async function toggleDebug() {
    await callApi('toggle_debug', 'toggleDebug');
    debugActive = !debugActive;
    const btn = document.getElementById('btn-debug');
    btn.textContent = debugActive ? 'Debug ●' : 'Debug';
    btn.classList.toggle('active', debugActive);
  }

  setInterval(refresh, 200);
</script>
</body>
</html>
"""


def launch_dashboard(session_kwargs):
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError("pywebview is not installed. Run: pip install pywebview") from exc

    state = SharedUIState()
    controller = SessionController(session_kwargs, state)
    api = DashboardAPI(state, controller)

    window = webview.create_window(
        "ASL Recognition",
        html=_render_html(),
        js_api=api,
        width=1100,
        height=720,
        min_size=(800, 500),
    )

    def on_closed():
        controller.stop()

    window.events.closed += on_closed
    webview.start()
