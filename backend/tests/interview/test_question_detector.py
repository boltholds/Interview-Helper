from app.interview.question_detector import QuestionDetector


def test_detects_explicit_question_once() -> None:
    detector = QuestionDetector()

    detected = detector.feed("Расскажите о проекте. Как вы обеспечивали отказоустойчивость?")

    assert detected is not None
    assert detected.text == "Как вы обеспечивали отказоустойчивость?"
    assert detected.confidence == 0.98
    assert detector.feed("Расскажите о проекте. Как вы обеспечивали отказоустойчивость?") is None


def test_detects_question_cue_without_question_mark() -> None:
    detector = QuestionDetector()

    detected = detector.feed("Можете объяснить разницу между процессом и потоком")

    assert detected is not None
    assert detected.reason == "interrogative_cue"
    assert detected.confidence == 0.74


def test_ignores_plain_statement() -> None:
    detector = QuestionDetector()

    assert detector.feed("Мы используем Python и PostgreSQL в основном сервисе") is None


def test_reset_allows_new_session_to_emit_same_question() -> None:
    detector = QuestionDetector()
    text = "Почему вы выбрали FastAPI?"

    assert detector.feed(text) is not None
    detector.reset()
    assert detector.feed(text) is not None
