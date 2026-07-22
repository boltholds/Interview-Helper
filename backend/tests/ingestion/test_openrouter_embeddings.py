import io
import json
import urllib.request

from ingestion.embeddings import OpenRouterEmbeddingProvider


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def test_openrouter_embeddings_use_provider_endpoint_and_input_types(monkeypatch) -> None:
    requests: list[tuple[urllib.request.Request, dict[str, object]]] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float):
        assert timeout == 60.0
        payload = json.loads(request.data or b"{}")
        requests.append((request, payload))
        count = len(payload["input"])
        response = {
            "data": [
                {"index": index, "embedding": [float(index + 1), 0.5]}
                for index in range(count)
            ]
        }
        return _Response(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenRouterEmbeddingProvider(
        api_key="test-key",
        model="qwen/test-embedding",
        http_referer="https://example.test",
        app_title="Interview Helper Tests",
    )

    documents = provider.embed_documents(["first", "second"])
    query = provider.embed_query("question")

    assert documents == [[1.0, 0.5], [2.0, 0.5]]
    assert query == [1.0, 0.5]
    assert requests[0][0].full_url == "https://openrouter.ai/api/v1/embeddings"
    assert requests[0][1]["input_type"] == "search_document"
    assert requests[1][1]["input_type"] == "search_query"
    assert requests[0][0].get_header("Authorization") == "Bearer test-key"
    assert requests[0][0].get_header("X-openrouter-title") == "Interview Helper Tests"
