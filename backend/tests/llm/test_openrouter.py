import asyncio
import json

import httpx

from llm.base import ChatMessage
from llm.openrouter import OpenRouterLLMProvider


def test_openrouter_streams_chat_completion_deltas() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["x-openrouter-title"] == "Interview Helper Tests"
        payload = json.loads(request.content)
        assert payload["model"] == "test/model"
        assert payload["stream"] is True
        content = (
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            ': OPENROUTER PROCESSING\n\n'
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            'data: [DONE]\n\n'
        )
        return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

    async def scenario() -> str:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://openrouter.ai/api/v1",
        ) as client:
            provider = OpenRouterLLMProvider(
                api_key="test-key",
                model="test/model",
                app_title="Interview Helper Tests",
                client=client,
            )
            result = await provider.complete(
                [
                    ChatMessage(role="system", content="Be concise."),
                    ChatMessage(role="user", content="Say hello."),
                ]
            )
            await provider.close()
            return result

    assert asyncio.run(scenario()) == "Hello world"
