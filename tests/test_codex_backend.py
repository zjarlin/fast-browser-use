"""验证 Codex 凭据复用、Responses 契约与不执行未知动作的边界。"""

import io
import json
import subprocess
import urllib.error
from unittest.mock import Mock

import pytest

from fast_browser_use import codex_backend, demo, model


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_PROVIDER_KEY", "test-only-key")
    monkeypatch.delenv("FBU_MODEL", raising=False)
    monkeypatch.delenv("FBU_REASONING_EFFORT", raising=False)
    (tmp_path / "config.toml").write_text('''
model = "cheap-model"
model_provider = "proxy"
model_reasoning_effort = "none"
[model_providers.proxy]
base_url = "http://127.0.0.1:9999/v1/"
wire_api = "responses"
env_key = "TEST_PROVIDER_KEY"
''')
    return codex_backend.CodexModel()


def response(value, **extra):
    payload = {
        "status": "completed", "model": "cheap-model",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}],
        "usage": {"input_tokens": 42, "output_tokens": 6, "input_tokens_details": {"cached_tokens": 10}},
        **extra,
    }
    return io.BytesIO(json.dumps(payload).encode())


def page():
    return {
        "title": "Search", "url": "http://example.test", "text": "City Search",
        "actions": [
            {"id": "e1", "kind": "click", "role": "button", "label": "Search", "node": 1},
        ],
    }


def test_reuses_provider_and_reports_real_usage_without_fabricated_probabilities(engine):
    engine.transport = Mock()
    engine.transport.open.return_value = response({"choice": "A"})
    result = engine.choose(page(), "Click Search", [])
    request = engine.transport.open.call_args.args[0]
    body = json.loads(request.data)
    assert request.full_url == "http://127.0.0.1:9999/v1/responses"
    assert request.get_header("Authorization") == "Bearer test-only-key"
    assert body["model"] == "cheap-model" and body["reasoning"] == {"effort": "none"}
    assert body["store"] is False and body["stream"] is False
    assert body["text"]["format"]["schema"]["properties"]["choice"]["enum"] == ["A", "B", "C"]
    assert result["choice"] == "e1" and result["operation"] == "CLICK"
    assert result["confidence"] is None and result["probabilities"]["e1"] is None
    assert result["usage"] == {"prompt_tokens": 42, "completion_tokens": 6, "cached_tokens": 10}
    assert "test-only-key" not in json.dumps(result)


@pytest.mark.parametrize("answer", [
    {"choice": "Z"}, {"choice": "A", "code": "run()"}, [], {"choice": 1}, {"B": "DONE"},
])
def test_invalid_structured_choice_never_becomes_an_action(engine, answer):
    engine.transport = Mock()
    engine.transport.open.return_value = response(answer)
    with pytest.raises(ValueError, match="unknown action"):
        engine.choose(page(), "Click Search", [])


@pytest.mark.parametrize("answer", [{"text": None}, {"text": ""}, {"text": "value", "extra": True}])
def test_missing_or_malformed_field_never_gets_typed(engine, answer):
    engine.transport = Mock()
    engine.transport.open.return_value = response(answer)
    with pytest.raises(ValueError):
        engine.generate_text({"goal": "Enter a city"})


def test_field_and_plan_are_generated_by_provider(engine):
    engine.transport = Mock()
    engine.transport.open.side_effect = [response({"text": "天津"}), response({"steps": ["选择天津"]})]
    assert engine.generate_text({"goal": "Enter 天津"})[0] == "天津"
    assert "天津".encode() in engine.transport.open.call_args.args[0].data
    assert engine.plan("选择天津", {"title": "City"})[0] == ["选择天津"]


def test_select_current_value_is_distinct_from_the_offered_target(engine):
    state = page()
    state["actions"].append({
        "id": "e2", "kind": "select", "node": 2, "role": "combobox", "label": "模式 → 节能",
        "value": "energy", "current_value": "标准",
    })
    engine.transport = Mock()
    engine.transport.open.return_value = response({"choice": "B"})
    assert engine.choose(state, "将模式从标准切换到节能", [])["choice"] == "e2"
    prompt = json.loads(engine.transport.open.call_args.args[0].data)["input"]
    context, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{"user_task"'):])
    select = next(control for control in context["controls"] if control["role"] == "combobox")
    assert select["value"] == "标准"
    assert select["options"][0]["value"] == "energy"


def test_auth_command_is_captured_and_reused_without_copying_to_disk(engine, monkeypatch):
    engine.provider["auth"] = {"command": "secret-helper", "args": ["read"], "timeout_ms": 2000}
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "fresh-test-token\n", ""))
    monkeypatch.setattr(codex_backend.subprocess, "run", run)
    assert engine.headers()["Authorization"] == "Bearer fresh-test-token"
    assert engine.headers()["Authorization"] == "Bearer fresh-test-token"
    assert run.call_count == 2
    assert run.call_args.kwargs["capture_output"] is True
    assert run.call_args.kwargs["timeout"] == 2
    run.return_value = subprocess.CompletedProcess([], 1, "secret", "secret error")
    with pytest.raises(RuntimeError, match="no usable credential") as error:
        engine.headers()
    assert "secret" not in str(error.value)
    assert not (engine.config_dir / "auth.json").exists()


def test_missing_configured_env_key_does_not_fall_back_to_another_credential(engine, monkeypatch):
    monkeypatch.delenv("TEST_PROVIDER_KEY")
    (engine.config_dir / "auth.json").write_text('{"OPENAI_API_KEY":"another-account"}')
    with pytest.raises(ValueError, match="environment variable is missing"):
        engine.headers()


def test_incomplete_and_http_error_do_not_expose_provider_body(engine):
    engine.transport = Mock()
    engine.transport.open.return_value = response({"choice": "A"}, status="incomplete")
    with pytest.raises(ValueError, match="did not complete"):
        engine.choose(page(), "Click Search", [])
    engine.transport.open.side_effect = urllib.error.HTTPError(engine.url, 401, "secret", {}, io.BytesIO(b"secret"))
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        engine.choose(page(), "Click Search", [])
    assert "secret" not in str(error.value)


def test_codex_selection_never_loads_local_weights(engine, monkeypatch):
    monkeypatch.setattr(model, "get_model", lambda: engine)
    engine.transport = Mock()
    engine.transport.open.return_value = response({"choice": "A"})
    assert model.choose(page(), "Click Search", [])["backend"] == "codex"


def test_transient_model_requests_retry_with_a_bound_and_report_attempts(engine, monkeypatch):
    monkeypatch.setattr(codex_backend.time, "sleep", Mock())
    engine.transport = Mock()
    error = urllib.error.HTTPError(engine.url, 502, "upstream unavailable", {}, io.BytesIO())
    engine.transport.open.side_effect = [error, error, response({"choice": "A"})]
    result = engine.choose(page(), "Click Search", [])
    assert result["choice"] == "e1" and result["request_attempts"] == 3
    assert engine.transport.open.call_count == 3
    engine.transport.open.reset_mock()
    engine.transport.open.side_effect = error
    with pytest.raises(RuntimeError, match="HTTP 502"):
        engine.choose(page(), "Click Search", [])
    assert engine.transport.open.call_count == 3


def test_user_defaults_allow_environment_override_and_keep_keys_out_of_fbu(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("FBU_MODEL", "explicit-model")
    monkeypatch.delenv("FBU_BACKEND", raising=False)
    folder = tmp_path / "fast-browser-use"
    folder.mkdir()
    (folder / "config.toml").write_text('backend="codex"\nmodel="cheap-model"\n')
    demo.load_environment()
    assert codex_backend.os.environ["FBU_BACKEND"] == "codex"
    assert codex_backend.os.environ["FBU_MODEL"] == "explicit-model"
