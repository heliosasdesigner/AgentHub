from __future__ import annotations

from dataclasses import dataclass

from ..config import settings


@dataclass
class LLMEndpoint:
    model: str
    api_key: str | None
    base_url: str | None


def pick_llm_model(requested_model: str | None = None) -> str:
    """
    Placeholder LLM router.

    Extend to choose between OpenAI/vLLM/LM Studio/etc. For now, prefer user
    supplied model name and fall back to a default.
    """
    if requested_model:
        return requested_model
    if settings.openai_api_key:
        return "gpt-3.5-turbo"
    if settings.lmstudio_model:
        return settings.lmstudio_model
    return settings.vllm_model


def resolve_llm_endpoint(
    requested_model: str | None = None,
    requested_base_url: str | None = None,
    requested_api_key: str | None = None,
) -> LLMEndpoint:
    """
    Choose an OpenAI-compatible endpoint, preferring an explicit request,
    then OpenAI, LM Studio, and finally the configured vLLM server.
    """
    model = pick_llm_model(requested_model)

    if requested_base_url or requested_api_key:
        return LLMEndpoint(model=model, api_key=requested_api_key, base_url=requested_base_url)

    if settings.openai_api_key:
        return LLMEndpoint(model=model, api_key=settings.openai_api_key, base_url=None)

    if settings.lmstudio_base_url:
        return LLMEndpoint(
            model=model,
            api_key=settings.lmstudio_api_key,
            base_url=settings.lmstudio_base_url,
        )

    return LLMEndpoint(model=model, api_key=settings.vllm_api_key, base_url=settings.vllm_base_url)
