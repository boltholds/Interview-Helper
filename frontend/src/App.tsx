import { useEffect, useState } from "react";

type BackendState = "checking" | "online" | "offline";

const apiUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export default function App() {
  const [backendState, setBackendState] = useState<BackendState>("checking");

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${apiUrl}/api/v1/health`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Backend healthcheck failed");
        setBackendState("online");
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setBackendState("offline");
      });

    return () => controller.abort();
  }, []);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">AI INTERVIEW COPILOT</p>
          <h1>Interview Helper</h1>
        </div>
        <div className={`connection connection--${backendState}`}>
          <span className="connection__dot" />
          Backend: {backendState}
        </div>
      </header>

      <section className="session-controls panel">
        <label>
          Целевая роль
          <select defaultValue="python">
            <option value="python">Python Developer</option>
            <option value="backend">Backend Developer</option>
            <option value="ai">AI Engineer</option>
          </select>
        </label>
        <button type="button" disabled={backendState !== "online"}>
          Начать интервью
        </button>
      </section>

      <section className="workspace">
        <article className="panel">
          <p className="panel__label">Транскрипция</p>
          <h2>Речь интервьюера</h2>
          <p className="placeholder">
            Здесь будут появляться частичные и финальные сегменты транскрипции.
          </p>
        </article>

        <article className="panel panel--accent">
          <p className="panel__label">Обнаруженный вопрос</p>
          <h2>Ожидание вопроса</h2>
          <p className="placeholder">
            Автоматическое определение вопроса и ручное подтверждение появятся в следующем срезе.
          </p>
          <button className="button-secondary" type="button" disabled>
            Сформировать ответ вручную
          </button>
        </article>

        <article className="panel answer-panel">
          <p className="panel__label">Подсказка</p>
          <h2>Тезисы и ответ</h2>
          <p className="placeholder">
            После подключения retrieval и LLM здесь появятся тезисы, готовый ответ и источники.
          </p>
        </article>
      </section>
    </main>
  );
}
