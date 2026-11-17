"""
Model registry: which chat models the assistant may use, loaded from a YAML file.

The browser only ever names an alias from this registry. Provider strings, base URLs and keys stay on
the server, so a client cannot point the assistant at an arbitrary endpoint.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TOOL_MODES = {"native", "text"}
TEXT_PROTOCOLS = {"lfm", "hermes"}

# ${VAR} / ${VAR:-default} anywhere in a string, or a whole value of "os.environ/VAR" (LiteLLM's own
# convention, accepted so configs copied from LiteLLM proxy docs work unchanged).
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_OS_ENVIRON = "os.environ/"


class ConfigError(ValueError):
    """The registry file is malformed. Raised at load time so a bad deploy fails loudly."""


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    label: str
    local: bool
    litellm_params: dict[str, Any]
    context_window: int = 8192
    max_tokens: int = 2048
    temperature: float = 0.1
    tool_mode: str | None = None
    text_protocol: str | None = None
    # Agent budget per question. Generous by default: tools are cheap reads of data the proxy already
    # sent, and a review legitimately reads and proposes across many fields.
    max_tool_calls: int = 16
    max_rounds: int = 8
    max_calls_per_round: int = 8
    # Env vars the entry references but which are unset. A model with any is listed as unconfigured
    # rather than failing the whole registry: a missing OpenAI key must not take local models down.
    missing_env: tuple[str, ...] = ()

    @property
    def configured(self) -> bool:
        return not self.missing_env

    @property
    def provider_model(self) -> str:
        return str(self.litellm_params.get("model", ""))

    @property
    def api_base(self) -> str | None:
        base = self.litellm_params.get("api_base")
        return str(base) if base else None

    def public(self) -> dict[str, Any]:
        """What a client may see: no base URL, no key, no provider string."""
        return {
            "alias": self.alias,
            "label": self.label,
            "local": self.local,
            "configured": self.configured,
            "context_window": self.context_window,
        }


@dataclass(frozen=True)
class Registry:
    models: dict[str, ModelSpec]
    default: str | None
    fallbacks: dict[str, list[str]] = field(default_factory=dict)
    local_only: bool = False

    def usable(self) -> list[ModelSpec]:
        """Models a request may select: configured, and local when local_only is set."""
        return [m for m in self.models.values() if m.configured and (m.local or not self.local_only)]

    @staticmethod
    def _tail(model: str | None) -> str:
        return (model or "").rsplit("/", 1)[-1].lower()

    def _is_model(self, spec: ModelSpec, response_model: str | None) -> bool:
        a, b = self._tail(spec.provider_model), self._tail(response_model)
        # Providers add a date or variant suffix to the id they report ("gpt-6-luna-pro-2026...").
        return bool(a and b and (a.startswith(b) or b.startswith(a)))

    def served_by(self, spec: ModelSpec, response_model: str | None) -> ModelSpec | None:
        """
        The registry model that really answered when it is not the one asked for (a configured fallback
        took over), else None. Unknown response ids count as "as asked": the notice is for a known switch.
        """
        if not response_model or self._is_model(spec, response_model):
            return None
        for other in self.models.values():
            if other.alias != spec.alias and self._is_model(other, response_model):
                return other
        return None

    def resolve(self, alias: str | None) -> ModelSpec:
        usable = {m.alias: m for m in self.usable()}
        name = alias or self.default
        if name and name in usable:
            return usable[name]
        if alias:
            raise KeyError(alias)
        if usable:
            return next(iter(usable.values()))
        raise KeyError("<default>")

    def router_fallbacks(self) -> list[dict[str, list[str]]]:
        usable = {m.alias for m in self.usable()}
        out = []
        for src, targets in self.fallbacks.items():
            kept = [t for t in targets if t in usable and t != src]
            if src in usable and kept:
                out.append({src: kept})
        return out


def expand_env(value: Any, env: Mapping[str, str], missing: set[str]) -> Any:
    """Expand env references recursively; record names that are unset and have no default."""
    if isinstance(value, dict):
        return {k: expand_env(v, env, missing) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, env, missing) for v in value]
    if not isinstance(value, str):
        return value
    if value.startswith(_OS_ENVIRON):
        name = value[len(_OS_ENVIRON):]
        if env.get(name):
            return env[name]
        missing.add(name)
        return ""

    def sub(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        if env.get(name):
            return env[name]
        if default is not None:
            return default
        missing.add(name)
        return ""

    return _ENV_RE.sub(sub, value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_registry(data: Mapping[str, Any], env: Mapping[str, str] | None = None) -> Registry:
    env = os.environ if env is None else env
    if not isinstance(data, Mapping):
        raise ConfigError("registry must be a mapping")
    entries = data.get("models")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("registry needs a non-empty 'models' list")

    models: dict[str, ModelSpec] = {}
    for i, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise ConfigError(f"models[{i}] must be a mapping")
        alias = str(raw.get("alias", ""))
        if not ALIAS_RE.match(alias):
            raise ConfigError(f"models[{i}]: invalid alias {alias!r}")
        if alias in models:
            raise ConfigError(f"duplicate alias {alias!r}")
        params = raw.get("litellm_params")
        if not isinstance(params, Mapping) or not params.get("model"):
            raise ConfigError(f"{alias}: litellm_params.model is required")
        if "local" not in raw:
            # Explicit on purpose: "local" decides whether record data may leave the server, so it is
            # never inferred from a hostname.
            raise ConfigError(f"{alias}: 'local: true|false' is required")
        tool_mode = raw.get("tool_mode")
        if tool_mode is not None and tool_mode not in TOOL_MODES:
            raise ConfigError(f"{alias}: tool_mode must be one of {sorted(TOOL_MODES)}")
        text_protocol = raw.get("text_protocol")
        if text_protocol is not None and text_protocol not in TEXT_PROTOCOLS:
            raise ConfigError(f"{alias}: text_protocol must be one of {sorted(TEXT_PROTOCOLS)}")

        missing: set[str] = set()
        expanded = expand_env(dict(params), env, missing)
        try:
            spec = ModelSpec(
                alias=alias,
                label=str(raw.get("label") or alias),
                local=_truthy(raw["local"]),
                litellm_params=expanded,
                context_window=int(raw.get("context_window", 8192)),
                max_tokens=int(raw.get("max_tokens", 2048)),
                temperature=float(raw.get("temperature", 0.1)),
                tool_mode=tool_mode,
                text_protocol=text_protocol,
                max_tool_calls=int(raw.get("max_tool_calls", 16)),
                max_rounds=int(raw.get("max_rounds", 8)),
                max_calls_per_round=int(raw.get("max_calls_per_round", 8)),
                missing_env=tuple(sorted(missing)),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{alias}: {exc}") from exc
        if spec.context_window < 1024 or spec.max_tokens < 1 or spec.max_tokens >= spec.context_window:
            raise ConfigError(f"{alias}: need context_window >= 1024 and 0 < max_tokens < context_window")
        if not (1 <= spec.max_tool_calls <= 40 and 1 <= spec.max_rounds <= 12 and 1 <= spec.max_calls_per_round <= spec.max_tool_calls):
            raise ConfigError(f"{alias}: need 1 <= max_tool_calls <= 40, 1 <= max_rounds <= 12, 1 <= max_calls_per_round <= max_tool_calls")
        models[alias] = spec

    default = data.get("default")
    if default is not None and default not in models:
        raise ConfigError(f"default {default!r} is not a declared alias")

    fallbacks_raw = data.get("fallbacks") or {}
    if not isinstance(fallbacks_raw, Mapping):
        raise ConfigError("'fallbacks' must map alias -> [aliases]")
    fallbacks: dict[str, list[str]] = {}
    for src, targets in fallbacks_raw.items():
        if src not in models or not isinstance(targets, list) or any(t not in models for t in targets):
            raise ConfigError(f"fallbacks for {src!r} reference an undeclared alias")
        if models[src].local and any(not models[t].local for t in targets):
            # Local means the data stays on this server; a fallback to a hosted model would quietly end that.
            raise ConfigError(f"local model {src!r} may not fall back to a hosted model")
        fallbacks[src] = list(targets)

    local_only = _truthy(env.get("ASSISTANT_LOCAL_ONLY", data.get("local_only", False)))
    return Registry(models=models, default=default, fallbacks=fallbacks, local_only=local_only)


def load_registry(path: Path, env: Mapping[str, str] | None = None) -> Registry:
    import yaml

    if not path.is_file():
        raise ConfigError(f"model registry not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: {exc}") from exc
    return parse_registry(data or {}, env)
