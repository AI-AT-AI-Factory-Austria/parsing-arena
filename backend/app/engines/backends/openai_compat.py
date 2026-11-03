"""Chat backend for an OpenAI-compatible server, e.g. vLLM as used by ParseBench."""

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


def chat_completions_url(endpoint_url: str) -> str:
    base = endpoint_url.rstrip("/")
    return base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"


class OpenAICompatBackend:
    """Same request, retry and failure semantics as ParseBench's `_nano_chat`:
    5xx/transport errors are retried three times with backoff, 4xx is not
    retried, and a failed request yields None (the element keeps empty content)."""

    def __init__(self, endpoint_url: str, *, api_key: str = "", timeout_s: float = 900.0):
        self.url = chat_completions_url(endpoint_url)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers)

    async def complete(self, payload: dict) -> str | None:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._client.post(
                    self.url,
                    json={**payload, "chat_template_kwargs": {"enable_thinking": False}},
                )
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"{resp.status_code}", request=resp.request, response=resp
                    )
                if resp.status_code >= 400:
                    logger.warning("4xx from endpoint (not retried): %s", resp.text[:200])
                    return None
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, TimeoutError, KeyError, ValueError) as e:
                last_exc = e
                await asyncio.sleep(min(10.0, 2.0 * (2**attempt)))
        logger.warning("stage request failed after retries: %s", last_exc)
        return None

    async def aclose(self) -> None:
        await self._client.aclose()
