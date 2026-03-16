import json
import sys
import threading
import urllib.request
from typing import Optional


class LLMRewriter:
    def __init__(
        self,
        enabled: bool = False,
        model: str = "qwen3:4b",
        interval_sec: float = 1.5,
        min_tokens: int = 2,
        timeout_sec: float = 45.0,
    ):
        self.enabled = enabled
        self.model = model
        self.interval_sec = float(interval_sec)
        self.min_tokens = int(min_tokens)
        self.timeout_sec = float(timeout_sec)

        self._last_request = ""
        self._last_sentence = ""
        self._last_launch_ts = 0.0
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _rewrite_once(self, text: str) -> str:
        prompt = (
            "Rewrite these ASL gloss tokens into one short natural English sentence.\n"
            "Rules:\n"
            "- Output only the sentence.\n"
            "- Do not explain.\n"
            "- Do not add bullet points.\n"
            "- Do not add extra facts.\n"
            "- If unclear, output exactly: UNCLEAR\n"
            f"Tokens: {text}\n"
            "Sentence:"
        )
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "num_predict": 20,
                "num_ctx": 256,
            },
        }
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        sentence = str(payload.get("response", "")).strip()
        if not sentence:
            return ""

        for line in sentence.splitlines():
            line = line.strip()
            if line:
                return line
        return ""

    def _run_worker(self, text: str) -> None:
        try:
            sentence = self._rewrite_once(text)
        except Exception as exc:
            print(f"[llm_rewrite] worker error: {exc}", file=sys.stderr)
            sentence = ""

        if sentence:
            with self._lock:
                self._last_sentence = sentence

    def rewrite_now(self, transcript_text: str) -> str:
        if not self.enabled:
            return ""

        text = transcript_text.strip()
        if not text:
            return ""

        if len(text.split()) < self.min_tokens:
            return ""

        try:
            sentence = self._rewrite_once(text)
        except Exception as exc:
            print(f"[llm_rewrite] rewrite_now error: {exc}", file=sys.stderr)
            sentence = ""

        if sentence:
            with self._lock:
                self._last_request = text
                self._last_sentence = sentence
        return sentence

    def maybe_rewrite(self, transcript_text: str, now_ts: float) -> str:
        if not self.enabled:
            return ""

        text = transcript_text.strip()
        if not text:
            return self._last_sentence

        if len(text.split()) < self.min_tokens:
            return self._last_sentence

        with self._lock:
            worker_alive = self._worker is not None and self._worker.is_alive()
            should_launch = (
                (not worker_alive)
                and (text != self._last_request)
                and ((now_ts - self._last_launch_ts) >= self.interval_sec)
            )

            if should_launch:
                self._last_request = text
                self._last_launch_ts = now_ts
                self._worker = threading.Thread(target=self._run_worker, args=(text,), daemon=True)
                self._worker.start()

            return self._last_sentence
