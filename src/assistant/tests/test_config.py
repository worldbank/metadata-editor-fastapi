import pytest

from src.assistant.core.config import ConfigError, expand_env, parse_registry


def registry(env=None, **top):
    data = {
        "default": "local",
        "models": [
            {"alias": "local", "local": True, "litellm_params": {"model": "ollama_chat/qwen3", "api_base": "${BASE:-http://127.0.0.1:11434}"}},
            {"alias": "hosted", "local": False, "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "${OPENAI_API_KEY}"}},
            {"alias": "hosted2", "local": False, "litellm_params": {"model": "anthropic/x", "api_key": "os.environ/ANTHROPIC_API_KEY"}},
        ],
        "fallbacks": {"hosted": ["local"], "hosted2": ["local", "hosted"]},
    }
    data.update(top)
    return parse_registry(data, env or {})


def test_env_defaults_and_missing_keys():
    reg = registry({"OPENAI_API_KEY": "sk-test"})
    assert reg.models["local"].api_base == "http://127.0.0.1:11434"
    assert reg.models["hosted"].litellm_params["api_key"] == "sk-test"
    assert reg.models["hosted"].configured
    assert reg.models["hosted2"].missing_env == ("ANTHROPIC_API_KEY",)
    assert [m.alias for m in reg.usable()] == ["local", "hosted"]


def test_missing_key_does_not_break_local_models():
    reg = registry({})
    assert [m.alias for m in reg.usable()] == ["local"]
    assert reg.resolve(None).alias == "local"


def test_local_only_hides_hosted_models_and_their_fallbacks():
    reg = registry({"OPENAI_API_KEY": "k", "ASSISTANT_LOCAL_ONLY": "1"})
    assert [m.alias for m in reg.usable()] == ["local"]
    assert reg.router_fallbacks() == []
    with pytest.raises(KeyError):
        reg.resolve("hosted")


def test_fallbacks_keep_only_usable_targets():
    reg = registry({"OPENAI_API_KEY": "k"})
    assert reg.router_fallbacks() == [{"hosted": ["local"]}]  # hosted2 has no key, so it is not usable


def test_resolve_unknown_alias():
    with pytest.raises(KeyError):
        registry({}).resolve("nope")


def test_public_view_has_no_endpoint_or_key():
    pub = registry({"OPENAI_API_KEY": "sk-secret"}).models["hosted"].public()
    assert "sk-secret" not in repr(pub)
    assert set(pub) == {"alias", "label", "local", "configured", "context_window"}


@pytest.mark.parametrize(
    "models, message",
    [
        ([{"alias": "a", "litellm_params": {"model": "x"}}], "local"),
        ([{"alias": "bad alias", "local": True, "litellm_params": {"model": "x"}}], "invalid alias"),
        ([{"alias": "a", "local": True, "litellm_params": {}}], "litellm_params.model"),
        ([{"alias": "a", "local": True, "tool_mode": "magic", "litellm_params": {"model": "x"}}], "tool_mode"),
        ([{"alias": "a", "local": True, "max_tokens": 9000, "context_window": 8192, "litellm_params": {"model": "x"}}], "max_tokens"),
        (
            [
                {"alias": "a", "local": True, "litellm_params": {"model": "x"}},
                {"alias": "a", "local": True, "litellm_params": {"model": "y"}},
            ],
            "duplicate",
        ),
    ],
)
def test_rejects_malformed_entries(models, message):
    with pytest.raises(ConfigError, match=message):
        parse_registry({"models": models}, {})


def test_rejects_unknown_default_and_fallback():
    m = [{"alias": "a", "local": True, "litellm_params": {"model": "x"}}]
    with pytest.raises(ConfigError, match="default"):
        parse_registry({"default": "b", "models": m}, {})
    with pytest.raises(ConfigError, match="fallbacks"):
        parse_registry({"fallbacks": {"a": ["b"]}, "models": m}, {})


def test_expand_env_nested():
    missing = set()
    out = expand_env({"a": ["${X}", "pre-${Y:-d}"], "b": 3}, {"X": "1"}, missing)
    assert out == {"a": ["1", "pre-d"], "b": 3}
    assert missing == set()


def test_shipped_example_registry_parses_with_lfm_default():
    from pathlib import Path

    from src.assistant.core.config import load_registry

    reg = load_registry(Path(__file__).resolve().parents[3] / "assistant.models.yaml.example", {})
    assert reg.default == "local-lfm"
    lfm = reg.models["local-lfm"]
    assert lfm.local and lfm.configured and lfm.tool_mode == "native" and lfm.text_protocol == "lfm"
    assert lfm.litellm_params["api_base"] == "http://127.0.0.1:8080/v1"
    assert reg.resolve(None).alias == "local-lfm"


def test_default_falls_back_to_a_usable_model_when_hosted_key_is_missing():
    reg = registry({}, default="hosted")
    # The configured default is unusable (no key); the local model answers instead of failing.
    assert reg.resolve(None).alias == "local"
    with pytest.raises(KeyError):
        reg.resolve("hosted")


def test_served_by_names_the_model_that_took_over_after_a_fallback():
    reg = registry({"OPENAI_API_KEY": "k"})
    hosted, local = reg.models["hosted"], reg.models["local"]
    # Answered by the model that was asked for (with or without a provider date suffix): no notice.
    assert reg.served_by(hosted, "gpt-4o-mini") is None
    assert reg.served_by(hosted, "gpt-4o-mini-2026-01-01") is None
    assert reg.served_by(hosted, None) is None
    # The hosted model failed and the local one answered.
    assert reg.served_by(hosted, "qwen3").alias == "local"
    # A response id nothing in the registry matches is not reported as a switch.
    assert reg.served_by(hosted, "some-unknown-model") is None
    assert reg.served_by(local, "qwen3") is None
