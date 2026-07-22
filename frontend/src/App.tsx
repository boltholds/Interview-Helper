import { useEffect, useRef, useState } from "react";

import { MicrophoneStreamer } from "./audio";

type BackendState = "checking" | "online" | "offline";
type SessionState = "idle" | "connecting" | "recording" | "stopping" | "error";

type ServerEvent = {
  type: string;
  sequence?: number;
  payload?: Record<string, unknown>;
};

const apiUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const wsBaseUrl = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000";

export default function App() {
  const [backendState, setBackendState] = useState<BackendState>("checking");
  const [sessionState, setSessionState] = useState<SessionState>("idle");
  const [role, setRole] = useState("Python Developer");
  const [language, setLanguage] = useState("auto");
  const [finalTranscript, setFinalTranscript] = useState("");
  const [partialTranscript, setPartialTranscript] = useState("");
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

    if (event.type === "stt_stopped") {
      activeRef.current = false;
      socketRef.current?.close();
      setSessionState("idle");
      return;
    }

    if (event.type === "error") {
      const message = payload.message;
      setErrorMessage(typeof message === "string" ? message : "Unknown backend error");
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
    sequenceRef.current = 0;
    reconnectAttemptRef.current = 0;
    sessionIdRef.current = crypto.randomUUID();
    activeRef.current = true;
    connectSocket();
  }

  async function stopInterview(): Promise<void> {
    setSessionState("stopping");
    activeRef.current = false;
    await microphoneRef.current?.stop();
    microphoneRef.current = null;

    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      sendClientEvent("stop_session");
      activeRef.current = false;
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

        <article className="panel panel--accent">
          <p className="panel__label">Обнаруженный вопрос</p>
          <h2>Ожидание вопроса</h2>
          <p className="placeholder">
            Определение вопроса будет подключено к финальным сегментам транскрипции в следующем срезе.
          </p>
          <button className="button-secondary" type="button" disabled>
            Сформировать ответ вручную
          </button>
        </article>

        <article className="panel answer-panel">
          <p className="panel__label">Подсказка</p>
          <h2>Тезисы и ответ</h2>
          <p className="placeholder">
            Hybrid RAG готов. Генерация ответа будет подключена после определения вопроса и профиля кандидата.
          </p>
        </article>
      </section>
    </main>
  );
}
