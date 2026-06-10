"""Singleton LLM client for the metadata reviewer (AutoGen chat completion client)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


def _reviewer_get(*keys: str, default: str | None = None) -> str | None:
    """
    Return the first non-empty env value among ``keys`` (in order).

    Use ``REVIEWER_*`` names first in ``reviewer.env``; optional legacy names
    (e.g. ``AZURE_OPENAI_ENDPOINT``) remain as fallbacks for older configs.
    """
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            return v
    return default


@lru_cache(maxsize=1)
def get_reviewer_model_client() -> Any:
    """
    Return the underlying model client used by MetadataReviewerCore.

    Uses the same factories as ai4data.metadata.reviewer.MetadataReviewerClient.
    """
    from ai4data.metadata.reviewer import MetadataReviewerClient

    provider = (_reviewer_get("REVIEWER_PROVIDER", default="openai") or "openai").lower()
    model = _reviewer_get("REVIEWER_MODEL", default="gpt-4o-mini") or "gpt-4o-mini"

    if provider == "openai":
        api_key = _reviewer_get("REVIEWER_OPENAI_API_KEY", "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "REVIEWER_OPENAI_API_KEY is not set (required for REVIEWER_PROVIDER=openai; "
                "legacy fallback: OPENAI_API_KEY)"
            )
        return MetadataReviewerClient.from_openai(model, api_key)._core.model_client

    if provider == "azure":
        endpoint = _reviewer_get("REVIEWER_AZURE_ENDPOINT", "AZURE_OPENAI_ENDPOINT")
        deployment = _reviewer_get("REVIEWER_AZURE_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT")
        api_version = _reviewer_get("REVIEWER_AZURE_API_VERSION", "AZURE_OPENAI_API_VERSION")
        if not all([endpoint, deployment, api_version]):
            raise RuntimeError(
                "Azure OpenAI: set REVIEWER_AZURE_ENDPOINT, REVIEWER_AZURE_DEPLOYMENT, "
                "REVIEWER_AZURE_API_VERSION (legacy: AZURE_OPENAI_* unprefixed names)."
            )

        tenant_id = _reviewer_get("REVIEWER_AZURE_TENANT_ID", "AZURE_AD_TENANT_ID")
        client_id = _reviewer_get("REVIEWER_AZURE_CLIENT_ID", "AZURE_AD_CLIENT_ID")
        client_secret = _reviewer_get("REVIEWER_AZURE_CLIENT_SECRET", "AZURE_AD_CLIENT_SECRET")
        token_scope = (
            _reviewer_get(
                "REVIEWER_AZURE_TOKEN_SCOPE",
                "AZURE_OPENAI_TOKEN_SCOPE",
                default="https://cognitiveservices.azure.com/.default",
            )
            or "https://cognitiveservices.azure.com/.default"
        )
        api_key = _reviewer_get("REVIEWER_AZURE_API_KEY", "AZURE_OPENAI_API_KEY")

        if tenant_id and client_id and client_secret:
            from azure.identity import ClientSecretCredential, get_bearer_token_provider

            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
            token_provider = get_bearer_token_provider(credential, token_scope)
            return MetadataReviewerClient.from_azure(
                model,
                endpoint,
                deployment,
                api_version,
                azure_ad_token_provider=token_provider,
            )._core.model_client

        if api_key:
            return MetadataReviewerClient.from_azure(
                model,
                endpoint,
                deployment,
                api_version,
                azure_ad_token=api_key,
            )._core.model_client

        raise RuntimeError(
            "Azure OpenAI auth: set REVIEWER_AZURE_API_KEY, or all of "
            "REVIEWER_AZURE_TENANT_ID, REVIEWER_AZURE_CLIENT_ID, REVIEWER_AZURE_CLIENT_SECRET "
            "(and optionally REVIEWER_AZURE_TOKEN_SCOPE). "
            "Legacy unprefixed AZURE_* / AZURE_AD_* names are still accepted."
        )

    if provider == "ollama":
        port_s = _reviewer_get("REVIEWER_OLLAMA_PORT", "OLLAMA_PORT", default="11434")
        port = int(port_s or "11434")
        return MetadataReviewerClient.from_ollama(model, port=port)._core.model_client

    if provider == "anthropic":
        api_key = _reviewer_get("REVIEWER_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "REVIEWER_ANTHROPIC_API_KEY is not set (required for REVIEWER_PROVIDER=anthropic; "
                "legacy fallback: ANTHROPIC_API_KEY)"
            )
        return MetadataReviewerClient.from_anthropic(model, api_key)._core.model_client

    raise RuntimeError(f"Unknown REVIEWER_PROVIDER={provider!r} (use openai, azure, ollama, anthropic)")
