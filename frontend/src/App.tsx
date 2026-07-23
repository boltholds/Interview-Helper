import { useEffect, useRef, useState } from "react";

import { MicrophoneStreamer } from "./audio";

type BackendState = "checking" | "online" | "offline";
type SessionState = "idle" | "connecting" | "recording" | "stopping" | "error";
type GenerationState = "idle" | "retrieving" | "generating" | "completed" | "error";

type ServerEvent = {
  type: string;
  sequence?: number;
  payload?: Record<string, unknown>;
};

type AnswerSource = {
  title: string;
  source_path: string;
  score: number;
};

const apiUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const wsBaseUrl = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000";

function parseSources(value: unknown): AnswerSource[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const source = item as Record<string, unknown>;
    if (typeof source.title !== "string" || typeof source.source_path !== "string") return [];
    return [
      {
        title: source.title,
        source_path: source.source_path,
        score: typeof source.score === "number" ? source.score : 0,
      },
    ];
  });
}

export default function App() {
  const [backendState, setBackendState] = useState<BackendState>("checking");
  const [sessionState, setSessionState] = useState<SessionState>("idle");
  const [generationState, setGenerationState] = useState<GenerationState>("idle");
  const [role, setRole] = useState("Python Developer");
  const [language, setLanguage] = useState("auto");
  const [finalTranscript, setFinalTranscript] = useState("");
  const [partialTranscript, setPartialTranscript] = useState("");
  const [questionDraft, setQuestionDraft] = useState("");
  const [questionConfidence, setQuestionConfidence] = useState<number | null>(null);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<AnswerSource[]>([]);
  const [retrievalWarning, setRetrievalWarning] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const socketRef = useRef<WebSocket | null>(null);
  const microphoneRef = useRef<MicrophoneStreamer | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const sequenceRef = useRef(0);
  const activeRef = useRef(false);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<number | null>(null);

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

  useEffect(() => {
    return () => {
      activeRef.current = false;
      socketRef.current?.close();
      void microphoneRef.current?.stop();
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }
    };
  }, []);

  function nextSequence(): number {
    const sequence = sequenceRef.current;
    sequenceRef.current += 1;
    return sequence;
  }

  function sendClientEvent(type: string, payload: Record<string, unknown> = {}): void {
    const socket = socketRef.current;
    const sessionId = sessionIdRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN || !sessionId) return;
    socket.send(
      JSON.stringify({
        type,
        session_id: sessionId,
        sequence: nextSequence(),
        payload,
        sent_at: new Date().toISOString(),
      }),
    );
  }

  async function startMicrophone(): Promise<void> {
    if (microphoneRef.current) return;
    const microphone = new MicrophoneStreamer();
    microphoneRef.current = microphone;
    await microphone.start((chunk) => {
      const socket = socketRef.current;
      if (socket?.readyState === WebSocket.OPEN) socket.send(chunk);
    });
  }

  function handleServerEvent(event: ServerEvent): void {
    const payload = event.payload ?? {};

    if (event.type === "stt_ready") {
      reconnectAttemptRef.current = 0;
      void startMicrophone()
        .then(() => setSessionState("recording"))
        .catch((error: unknown) => {
          const message = error instanceof Error ? error.message : "Microphone access failed";
          setErrorMessage(message);
          setSessionState("error");
          activeRef.current = false;
          socketRef.current?.close();
        });
      return;
    }

    if (event.type === "transcript_partial") {
      const fullText = payload.full_text;
      if (typeof fullText === "string") setPartialTranscript(fullText);
      return;
    }

    if (event.type === "transcript_final") {
      const fullText = payload.full_text;
      if (typeof fullText === "string") setFinalTranscript(fullText);
      setPartialTranscript("");
      return;
    }

    if (event.type === "question_detected") {
      const question = payload.question;
      if (typeof question === "string") setQuestionDraft(question);
      setQuestionConfidence(typeof payload.confidence === "number" ? payload.confidence : null);
      return;
    }

    if (event.type === "generation_started") {
      setGenerationState("retrieving");
      setAnswer("");
      setSources(parseSources(payload.sources));
      setRetrievalWarning(
        typeof payload.retrieval_warning === "string" ? payload.retrieval_warning : "",
      );
      return;
    }

    if (event.type === "answer_delta") {
      const text = payload.text;
      setGenerationState("generating");
      if (typeof text === "string") setAnswer(text);
      return;
    }

    if (event.type === "answer_completed") {
      const completedAnswer = payload.answer;
      if (typeof completedAnswer === "string") setAnswer(completedAnswer);
      setSources(parseSources(payload.sources));
      setGenerationState("completed");
      return;
    }

    if (event.type === "stt_stopped") {
      activeRef.current = false;
      socketRef.current?.close();
      setSessionState("idle");
      return;
    }

    if (event.type === "error") {
      const message = payload.message;
      setErrorMessage(typeof message === "string" ? message : "Unknown backend error");
      if (payload.code === "generation_failed") setGenerationState("error");
      if (payload.code === "stt_unavailable") {
        activeRef.current = false;
        setSessionState("error");
        void microphoneRef.current?.stop();
        microphoneRef.current = null;
      }
    }
  }

  function connectSocket(): void {
    const sessionId = sessionIdRef.current;
    if (!sessionId || !activeRef.current) return;

    setSessionState("connecting");
    const socket = new WebSocket(`${wsBaseUrl}/ws/interview/${sessionId}`);
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;

    socket.onopen = () => {
      sendClientEvent("start_session", { language, role });
    };
    socket.onmessage = (message) => {
      if (typeof message.data !== "string") return;
      try {
        handleServerEvent(JSON.parse(message.data) as ServerEvent);
      } catch {
        setErrorMessage("Backend returned an invalid WebSocket event");
      }
    };
    socket.onerror = () => {
      setErrorMessage("WebSocket connection failed");
    };
    socket.onclose = () => {
      if (!activeRef.current) return;
      if (reconnectAttemptRef.current >= 3) {
        setSessionState("error");
        setErrorMessage("Не удалось восстановить соединение с backend");
        activeRef.current = false;
        void microphoneRef.current?.stop();
        microphoneRef.current = null;
        return;
      }
      const delay = 500 * 2 ** reconnectAttemptRef.current;
      reconnectAttemptRef.current += 1;
      reconnectTimerRef.current = window.setTimeout(connectSocket, delay);
    };
  }

  function startInterview(): void {
    setErrorMessage("");
    setFinalTranscript("");
    setPartialTranscript("");
    setQuestionDraft("");
    setQuestionConfidence(null);
    setAnswer("");
    setSources([]);
    setRetrievalWarning("");
    setGenerationState("idle");
    sequenceRef.current = 0;
    reconnectAttemptRef.current = 0;
    sessionIdRef.current = crypto.randomUUID();
    activeRef.current = true;
    connectSocket();
  }

  function generateAnswer(): void {
    const question = questionDraft.trim();
    if (!question) return;
    setErrorMessage("");
    setGenerationState("retrieving");
    sendClientEvent("commit_question", { question });
  }

  function cancelGeneration(): void {
    sendClientEvent("cancel_generation");
    setGenerationState("idle");
  }

  async function stopInterview(): Promise<void> {
    setSessionState("stopping");
    activeRef.current = false;
    await microphoneRef.current?.stop();
    microphoneRef.current = null;

    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      sendClientEvent("stop_session");
      window.setTimeout(() => {
        socket.close();
        setSessionState("idle");
      }, 5_000);
    } else {
      socket?.close();
      setSessionState("idle");
    }
  }

  const visibleTranscript = partialTranscript || finalTranscript;
  const sessionActive = ["connecting", "recording", "stopping"].includes(sessionState);
  const generationActive = ["retrieving", "generating"].includes(generationState);

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
          <select value={role} onChange={(event) => setRole(event.target.value)} disabled={sessionActive}>
            <option>Python Developer</option>
            <option>Backend Developer</option>
            <option>AI Engineer</option>
          </select>
        </label>
        <label>
          Язык
          <select
            value={language}
            onChange={(event) => setLanguage(event.target.value)}
            disabled={sessionActive}
          >
            <option value="auto">Авто</option>
            <option value="ru">Русский</option>
            <option value="en">English</option>
          </select>
        </label>
        {sessionActive ? (
          <button type="button" onClick={() => void stopInterview()} disabled={sessionState === "stopping"}>
            {sessionState === "stopping" ? "Завершение…" : "Остановить"}
          </button>
        ) : (
          <button type="button" onClick={startInterview} disabled={backendState !== "online"}>
            Начать интервью
          </button>
        )}
      </section>

      {errorMessage && <p className="error-banner">{errorMessage}</p>}

      <section className="workspace">
        <article className="panel transcript-panel">
          <div className="panel-heading">
            <div>
              <p className="panel__label">Транскрипция · whisper.cpp</p>
              <h2>Речь интервьюера</h2>
            </div>
            <span className={`recording-state recording-state--${sessionState}`}>{sessionState}</span>
          </div>
          <p className={visibleTranscript ? "transcript" : "placeholder"}>
            {visibleTranscript || "Нажмите «Начать интервью» и разрешите доступ к микрофону."}
          </p>
        </article>

        <article className="panel panel--accent question-panel">
          <div className="panel-heading">
            <div>
              <p className="panel__label">Обнаруженный вопрос</p>
              <h2>{questionDraft ? "Проверьте формулировку" : "Ожидание вопроса"}</h2>
            </div>
            {questionConfidence !== null && (
              <span className="confidence">{Math.round(questionConfidence * 100)}%</span>
            )}
          </div>
          <textarea
            value={questionDraft}
            onChange={(event) => setQuestionDraft(event.target.value)}
            placeholder="Вопрос появится автоматически. Его можно исправить перед генерацией."
            rows={5}
          />
          <div className="question-actions">
            <button
              type="button"
              onClick={generateAnswer}
              disabled={!questionDraft.trim() || generationActive || socketRef.current?.readyState !== WebSocket.OPEN}
            >
              {generationActive ? "Формируем ответ…" : "Сформировать ответ"}
            </button>
            {generationActive && (
              <button className="button-secondary" type="button" onClick={cancelGeneration}>
                Отменить
              </button>
            )}
          </div>
        </article>

        <article className="panel answer-panel">
          <div className="panel-heading">
            <div>
              <p className="panel__label">Подсказка · Hybrid RAG + OpenRouter</p>
              <h2>Тезисы и ответ</h2>
            </div>
            <span className={`generation-state generation-state--${generationState}`}>
              {generationState}
            </span>
          </div>
          {retrievalWarning && (
            <p className="warning-banner">RAG недоступен, используется ответ без локального контекста: {retrievalWarning}</p>
          )}
          <div className={answer ? "answer" : "placeholder"}>
            {answer || "Подтвердите обнаруженный вопрос, чтобы получить потоковую подсказку."}
          </div>
          {sources.length > 0 && (
            <div className="sources">
              <p className="panel__label">Использованные источники</p>
              <ul>
                {sources.map((source) => (
                  <li key={`${source.source_path}-${source.title}`}>
                    <strong>{source.title}</strong>
                    <span>{source.source_path} · score {source.score.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </article>
      </section>
    </main>
  );
}
