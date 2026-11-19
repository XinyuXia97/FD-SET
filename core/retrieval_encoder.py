from __future__ import annotations

import math
from typing import Any, Protocol


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class EmbeddingUnavailable(RuntimeError):
    """Raised when the configured embedding backend is not installed or usable."""


class ClipTextEncoder:
    """Frozen CLIP text encoder used by the paper's retrieval protocol.

    The import is intentionally lazy so the lightweight rule-based pipeline and
    unit tests do not require downloading a several-hundred-MB model.  The
    preferred implementation is Hugging Face Transformers; an installation of
    OpenAI's ``clip`` package is accepted as a fallback.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise EmbeddingUnavailable("CLIP retrieval requires torch") from exc

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._device = torch.device(resolved_device)
        self._backend = ""

        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore

            load_kwargs: dict[str, Any] = {}
            if cache_dir:
                load_kwargs["cache_dir"] = cache_dir
            if local_files_only:
                load_kwargs["local_files_only"] = True
            self._processor = CLIPProcessor.from_pretrained(model_name, **load_kwargs)
            self._model = CLIPModel.from_pretrained(model_name, **load_kwargs)
            self._model.to(self._device)
            self._model.eval()
            self._backend = "transformers"
            return
        except (ImportError, OSError, RuntimeError) as exc:
            transformers_error = exc

        try:
            import clip  # type: ignore

            if not hasattr(clip, "load"):
                raise ImportError("the local `clip` module does not expose OpenAI CLIP's load API")
            self._model, _ = clip.load(model_name, device=str(self._device), download_root=cache_dir)
            self._model.eval()
            self._processor = clip.tokenize
            self._backend = "openai_clip"
        except (AttributeError, ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - environment dependent
            raise EmbeddingUnavailable(
                "No usable CLIP backend found. Install `transformers` or `openai-clip`, "
                f"and make `{model_name}` available locally. Transformers error: {transformers_error}; "
                f"OpenAI CLIP error: {exc}"
            ) from exc

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._torch.no_grad():
            if self._backend == "transformers":
                inputs = self._processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                features = self._model.get_text_features(**inputs)
            else:
                tokens = self._processor(texts).to(self._device)
                features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return features.detach().cpu().tolist()


_ENCODER_CACHE: dict[tuple[str, str, str], TextEncoder] = {}


def build_text_encoder(config: dict[str, Any]) -> TextEncoder:
    backend = str(config.get("backend", "lexical")).strip().lower()
    if backend not in {"clip", "embedding", "hybrid"}:
        raise EmbeddingUnavailable(f"Embedding backend is not enabled: {backend}")
    model_name = str(config.get("model_name", "openai/clip-vit-base-patch32")).strip()
    device = str(config.get("device", "")).strip()
    cache_dir = str(config.get("cache_dir", "")).strip()
    local_files_only = bool(config.get("local_files_only", False))
    cache_key = (backend, model_name, f"{device}|{cache_dir}|{local_files_only}")
    if cache_key not in _ENCODER_CACHE:
        _ENCODER_CACHE[cache_key] = ClipTextEncoder(
            model_name,
            device=device or None,
            cache_dir=cache_dir or None,
            local_files_only=local_files_only,
        )
    return _ENCODER_CACHE[cache_key]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(float(left[idx]) * float(right[idx]) for idx in range(size))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left[:size]))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right[:size]))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return dot / (left_norm * right_norm)
