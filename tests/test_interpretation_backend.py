from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from core.failure_interpret import interpret_failure
from core.interpretation_backend import call_model_interpreter
from core.models import FailureEvent


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._buffer = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def read(self) -> bytes:
        return self._buffer.read()

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_openai_compatible_backend_returns_model_output(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Return JSON only.", encoding="utf-8")
    frame_path = tmp_path / "frame_000.png"
    Image.new("RGB", (16, 16), color=(32, 64, 96)).save(frame_path)

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == "test-model"
        assert "think" not in payload
        user_content = payload["messages"][1]["content"]
        assert isinstance(user_content, list)
        assert user_content[0]["type"] == "text"
        assert user_content[1]["type"] == "image_url"
        assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "failure_type": "collision",
                                    "interaction_pattern": "adversarial_merge",
                                    "key_entities": ["ego", "vehicle"],
                                    "risk_cues": ["short_ttc"],
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    event = FailureEvent(
        route_id="route_001",
        timestamp_s=1.0,
        failure_type="collision",
        ego_speed=8.0,
        description="cut-in vehicle merged aggressively",
        scene_tags=["urban", "merge"],
        risk_level="high",
        metadata={
            "context_summary": {
                "evidence_frames": [str(frame_path)],
            }
        },
    )

    semantics = interpret_failure(
        event,
        prompt_path=prompt_path,
        config={
            "backend": "openai_compatible",
            "model": "test-model",
            "api_key_env": "OPENAI_API_KEY",
            "fallback_to_rule_based": False,
        },
    )

    assert semantics["interaction_pattern"] == "adversarial_merge"
    assert semantics["interpreter"]["backend"] == "openai_compatible"


def test_openai_compatible_backend_supports_anonymous_local_service(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Return JSON only.", encoding="utf-8")

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        assert "Authorization" not in request.headers
        return _FakeResponse(
            {"choices": [{"message": {"content": '{"interaction_pattern":"control_instability"}'}}]}
        )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    event = FailureEvent("route", 1.0, "offroad", 3.0, "lane departure", ["urban"], "medium")

    semantics = interpret_failure(
        event,
        prompt_path=prompt_path,
        config={
            "backend": "openai_compatible",
            "model": "Qwen/Qwen-VL-Chat",
            "allow_anonymous": True,
            "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
            "fallback_to_rule_based": False,
        },
    )

    assert semantics["interaction_pattern"] == "control_instability"


def test_ollama_backend_uses_native_vision_payload(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Return JSON only.", encoding="utf-8")
    frame_path = tmp_path / "frame_001.png"
    Image.new("RGB", (16, 16), color=(128, 32, 16)).save(frame_path)

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == "qwen3-vl:8b-instruct"
        assert payload["think"] is False
        assert payload["format"] == "json"
        assert payload["messages"][1]["images"]
        return _FakeResponse(
            {"message": {"content": '{"interaction_pattern":"adversarial_merge"}'}}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    event = FailureEvent(
        "route", 1.0, "collision", 3.0, "cut-in vehicle", ["merge"], "high",
        {"context_summary": {"evidence_frames": [str(frame_path)]}},
    )
    semantics = call_model_interpreter(
        event,
        prompt_template=prompt_path.read_text(encoding="utf-8"),
        config={
            "backend": "ollama",
            "model": "qwen3-vl:8b-instruct",
            "endpoint": "http://127.0.0.1:11434/api/chat",
            "think": False,
            "json_mode": True,
        },
    )

    assert semantics["interaction_pattern"] == "adversarial_merge"


def test_ollama_rejects_unfinished_thinking_instead_of_fabricating_visual_result(
    tmp_path: Path, monkeypatch,
) -> None:
    frame = tmp_path / "actual_frame.png"
    Image.new("RGB", (16, 16), color=(32, 64, 96)).save(frame)
    requests = []

    def unfinished_reply(request, timeout=0):
        requests.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse({
            "message": {
                "content": "",
                "thinking": '{"interaction_pattern":"adversarial_merge"}',
            },
            "done_reason": "length",
            "eval_count": 8192,
        })

    monkeypatch.setattr("urllib.request.urlopen", unfinished_reply)
    event = FailureEvent(
        "RouteScenario_1773_rep0", 200.05, "low_speed", 1.0, "slow", ["urban"], "high",
        {"context_summary": {"evidence_frames": [str(frame)]}},
    )
    with pytest.raises(ValueError, match=r"no final JSON for RouteScenario_1773_rep0.*done_reason=length"):
        call_model_interpreter(
            event, prompt_template="Return JSON only.",
            config={"backend": "ollama", "model": "qwen3-vl:8b-instruct", "require_evidence_frames": True},
        )
    assert len(requests) == 1


def test_ollama_paper_mode_refuses_text_only_failure_interpretation(monkeypatch) -> None:
    def no_request(*args, **kwargs):
        raise AssertionError("text-only inference must never contact a vision service")

    monkeypatch.setattr("urllib.request.urlopen", no_request)
    event = FailureEvent("route_1711", 1.0, "collision", 3.0, "cut-in", ["merge"], "high")
    with pytest.raises(ValueError, match="requires real evidence frames"):
        call_model_interpreter(
            event,
            prompt_template="Return JSON describing observed interaction.",
            config={"backend": "ollama", "model": "qwen3-vl:8b-instruct", "require_evidence_frames": True},
        )


def test_paper_schema_rejects_incomplete_and_unknown_interactions(tmp_path: Path, monkeypatch) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return JSON only.", encoding="utf-8")
    event = FailureEvent("route", 1.0, "collision", 0.0, "cut-in", ["merge"], "high")
    config = {"backend": "ollama", "strict_schema": True, "fallback_to_rule_based": False}
    monkeypatch.setattr("core.failure_interpret.call_model_interpreter",
                        lambda *args, **kwargs: {"interaction_pattern": "adversarial_merge"})
    with pytest.raises(ValueError, match="missing="):
        interpret_failure(event, prompt_path=prompt, config=config)

    complete = {
        "failure_type": "collision", "interaction_pattern": "unlisted_action",
        "key_entities": ["ego", "vehicle"], "risk_cues": [],
        "capability_weakness": "merging", "spatial_pattern": "right lane",
        "temporal_pattern": "late merge", "risk_pattern": "short gap",
        "road_topology": "highway", "traffic_state": "moving",
        "relative_relations": [], "context_conditions": [],
    }
    monkeypatch.setattr("core.failure_interpret.call_model_interpreter",
                        lambda *args, **kwargs: complete)
    with pytest.raises(ValueError, match="Unknown VLM interaction_pattern"):
        interpret_failure(event, prompt_path=prompt, config=config)
    complete["interaction_pattern"] = "blocked_progress"
    assert interpret_failure(event, prompt_path=prompt, config=config)["interaction_pattern"] == "late_response"


def test_paper_schema_rejects_wrapped_json() -> None:
    from core.interpretation_backend import _parse_json_object

    with pytest.raises(ValueError, match="strict JSON object"):
        _parse_json_object('```json\n{"interaction_pattern":"late_response"}\n```', strict=True)
