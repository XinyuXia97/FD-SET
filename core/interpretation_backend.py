from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .models import FailureEvent


DEFAULT_OPENAI_COMPATIBLE_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def call_model_interpreter(
    event: FailureEvent,
    *,
    prompt_template: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    backend = str(config.get("backend", "rule_based")).strip().lower()
    if backend == "ollama":
        return _call_ollama_backend(event, prompt_template=prompt_template, config=config)
    if backend != "openai_compatible":
        raise ValueError(f"Unsupported interpretation backend: {backend}")

    return _call_openai_compatible_backend(event, prompt_template=prompt_template, config=config)


def _call_openai_compatible_backend(
    event: FailureEvent,
    *,
    prompt_template: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    import os

    model = str(config.get("model", "")).strip()
    if not model:
        raise ValueError("interpretation.model is required for backend=openai_compatible")

    api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.environ.get(api_key_env, "").strip()
    endpoint = str(config.get("endpoint", DEFAULT_OPENAI_COMPATIBLE_ENDPOINT)).strip()
    if not api_key and not bool(config.get("allow_anonymous", False)):
        raise RuntimeError(f"Environment variable {api_key_env} is required for backend=openai_compatible")

    timeout_s = float(config.get("timeout_s", 30.0))
    temperature = float(config.get("temperature", 0.0))

    request_payload = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt_template},
            {
                "role": "user",
                "content": _build_user_message(event, config=config),
            },
        ],
    }
    max_tokens = config.get("max_tokens")
    if max_tokens is not None:
        request_payload["max_tokens"] = int(max_tokens)
    if "think" in config:
        # Ollama's Qwen3 models expose this switch to keep chain-of-thought
        # out of the content field used by the JSON parser.
        request_payload["think"] = bool(config["think"])

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Interpretation backend request failed: {exc}; response={details[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Interpretation backend request failed: {exc}") from exc

    content = _extract_openai_compatible_content(payload)
    semantics = _parse_json_object(content, strict=bool(config.get("strict_schema", False)))
    if not isinstance(semantics, dict):
        raise ValueError("Interpretation backend did not return a JSON object")
    return semantics


def _call_ollama_backend(
    event: FailureEvent,
    *,
    prompt_template: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Call Ollama's vision API and accept only finalized JSON model output."""
    model = str(config.get("model", "")).strip()
    if not model:
        raise ValueError("interpretation.model is required for backend=ollama")

    endpoint = str(config.get("endpoint", "http://127.0.0.1:11434/api/chat")).strip()
    temperature = float(config.get("temperature", 0.0))
    request_payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": bool(config.get("think", False)),
        "messages": [
            {"role": "system", "content": prompt_template},
            _build_ollama_user_message(event, config=config),
        ],
    }
    if bool(config.get("require_evidence_frames", False)) and not request_payload["messages"][1].get("images"):
        raise ValueError(
            f"Ollama vision interpretation requires real evidence frames for {event.route_id}; "
            "inspect failure_extract.evidence_root_template and agent sensor capture"
        )
    if bool(config.get("json_mode", True)):
        request_payload["format"] = "json"
    options: dict[str, Any] = {"temperature": temperature}
    max_tokens = config.get("max_tokens")
    if max_tokens is not None:
        options["num_predict"] = int(max_tokens)
    num_ctx = config.get("num_ctx")
    if num_ctx is not None:
        options["num_ctx"] = int(num_ctx)
    request_payload["options"] = options

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(config.get("timeout_s", 180.0))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed: {exc}; response={details[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    message = payload.get("message", {})
    if not isinstance(message, dict):
        raise ValueError("Ollama response missing message object")
    content = str(message.get("content", "")).strip()
    try:
        return _parse_json_object(content, strict=bool(config.get("strict_schema", False)))
    except ValueError as exc:
        result = "no final JSON" if not content else "invalid final JSON"
        raise ValueError(
            f"Ollama returned {result} for {event.route_id} at {event.timestamp_s:.3f}s "
            f"(done_reason={payload.get('done_reason', 'unknown')}, eval_count={payload.get('eval_count', 'unknown')}); "
            "refusing to use unfinished model thinking"
        ) from exc


def _build_user_message(event: FailureEvent, *, config: dict[str, Any]) -> str | list[dict[str, Any]]:
    text_payload = _build_event_text(event)
    if not bool(config.get("include_evidence_frames", True)):
        return text_payload

    image_items = _build_image_content_items(event, config=config)
    if not image_items:
        return text_payload

    return [{"type": "text", "text": text_payload}, *image_items]


def _build_ollama_user_message(event: FailureEvent, *, config: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "user", "content": _build_event_text(event)}
    image_items = _build_image_content_items(event, config=config)
    images: list[str] = []
    for item in image_items:
        image_url = item.get("image_url", {})
        if not isinstance(image_url, dict):
            continue
        url = str(image_url.get("url", ""))
        if "," in url:
            images.append(url.split(",", 1)[1])
    if images:
        message["images"] = images
    return message


def _build_event_text(event: FailureEvent) -> str:
    event_payload = {
        "route_id": event.route_id,
        "timestamp_s": event.timestamp_s,
        "failure_type": event.failure_type,
        "ego_speed": event.ego_speed,
        "description": event.description,
        "scene_tags": event.scene_tags,
        "risk_level": event.risk_level,
        "metadata": event.metadata,
    }
    return (
        "Interpret the following autonomous-driving failure event and return JSON only.\n\n"
        f"{json.dumps(event_payload, ensure_ascii=True, indent=2)}"
    )


def _build_image_content_items(event: FailureEvent, *, config: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = event.metadata or {}
    context_summary = metadata.get("context_summary", {})
    if not isinstance(context_summary, dict):
        return []

    raw_frames = context_summary.get("evidence_frames", [])
    if not isinstance(raw_frames, list) or not raw_frames:
        return []

    image_detail = str(config.get("image_detail", "low")).strip() or "low"
    max_images = int(config.get("max_images", 3))
    inline_local_images = bool(config.get("inline_local_images", True))
    max_image_edge = int(config.get("max_image_edge", 1024))
    image_quality = int(config.get("image_quality", 80))

    items: list[dict[str, Any]] = []
    for raw_frame in raw_frames[:max_images]:
        frame_path = Path(str(raw_frame))
        if not frame_path.exists() or not frame_path.is_file():
            continue
        if inline_local_images:
            image_url = _build_data_url(
                frame_path,
                max_image_edge=max_image_edge,
                image_quality=image_quality,
            )
        else:
            image_url = frame_path.as_uri()
        items.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": image_detail,
                },
            }
        )
    return items


def _build_data_url(path: Path, *, max_image_edge: int = 1024, image_quality: int = 80) -> str:
    if max_image_edge > 0:
        try:
            import io

            from PIL import Image  # type: ignore

            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail(
                    (max_image_edge, max_image_edge),
                    getattr(Image, "Resampling", Image).LANCZOS,
                )
                encoded = io.BytesIO()
                image.save(
                    encoded,
                    format="JPEG",
                    quality=max(1, min(100, image_quality)),
                    optimize=True,
                )
            return "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")
        except (ImportError, OSError, ValueError):
            # Keep the backend usable in minimal environments; the caller can
            # still send the original bytes when Pillow is unavailable.
            pass

    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _extract_openai_compatible_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Interpretation backend response missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("Interpretation backend response has invalid choice payload")

    message = first_choice.get("message", {})
    if not isinstance(message, dict):
        raise ValueError("Interpretation backend response missing message")

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        if text_parts:
            return "\n".join(text_parts)

    raise ValueError("Interpretation backend response missing textual content")


def _parse_json_object(content: str, *, strict: bool = False) -> dict[str, Any]:
    """Parse a JSON object; paper mode requires the entire reply to be one object."""
    text = str(content or "").strip()
    if not text:
        raise ValueError("Interpretation backend returned empty content")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    if strict:
        raise ValueError("Interpretation backend did not return a strict JSON object")
    if "```" in text:
        for block in text.split("```")[1::2]:
            candidate = block.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].lstrip("\\n ")
            try:
                value = json.loads(candidate)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue

    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
    raise ValueError("Interpretation backend did not return a parseable JSON object")


def validate_interpretation_config(config: dict[str, Any], *, prompt_path: Path) -> None:
    backend = str(config.get("backend", "rule_based")).strip().lower()
    if backend == "rule_based":
        return

    if backend not in {"openai_compatible", "ollama"}:
        raise ValueError("config.interpretation.backend must be `rule_based`, `openai_compatible`, or `ollama`")

    if not prompt_path.exists():
        raise FileNotFoundError(f"Interpretation prompt not found: {prompt_path}")

    if not str(config.get("model", "")).strip():
        raise ValueError("config.interpretation.model is required for backend=openai_compatible")
