"""Minimal Ollama client: JSON-schema constrained chat, plain chat, embeddings."""
import json
import requests
from . import config


class LLMError(RuntimeError):
    pass


def _post(path, payload):
    try:
        r = requests.post(f"{config.OLLAMA_URL}{path}", json=payload, timeout=config.LLM_TIMEOUT)
    except requests.RequestException as e:
        raise LLMError(f"Ollama not reachable at {config.OLLAMA_URL}: {e}")
    if r.status_code >= 300:
        raise LLMError(f"Ollama {path} failed: {r.status_code} {r.text[:300]}")
    return r.json()


def chat_text(system, user, model=None, temperature=0.2):
    out = _post("/api/chat", {
        "model": model or config.LLM_MODEL, "stream": False,
        "options": {"temperature": temperature},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    })
    return out["message"]["content"].strip()


def chat_json(system, user, schema, model=None):
    """Structured output: Ollama constrains decoding to the given JSON schema (Ollama >= 0.5)."""
    out = _post("/api/chat", {
        "model": model or config.LLM_MODEL, "stream": False, "format": schema,
        "options": {"temperature": 0},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    })
    text = out["message"]["content"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise LLMError(f"model did not return valid JSON: {text[:300]}")


def chat_vision(system, user, images, schema=None, model=None, temperature=0):
    """Vision-language chat: `images` are file paths (PNG/JPG). Returns parsed JSON if `schema`, else text."""
    import base64
    from pathlib import Path
    b64 = [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in images]
    payload = {
        "model": model or config.VISION_MODEL, "stream": False,
        "options": {"temperature": temperature},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user, "images": b64}],
    }
    if schema:
        payload["format"] = schema
    text = _post("/api/chat", payload)["message"]["content"]
    if not schema:
        return text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise LLMError(f"vision model did not return valid JSON: {text[:300]}")


def embed(texts, model=None):
    """Return a list of embedding vectors (batched)."""
    if isinstance(texts, str):
        texts = [texts]
    vecs = []
    for i in range(0, len(texts), 64):
        out = _post("/api/embed", {"model": model or config.EMBED_MODEL, "input": texts[i:i + 64]})
        vecs.extend(out["embeddings"])
    return vecs
