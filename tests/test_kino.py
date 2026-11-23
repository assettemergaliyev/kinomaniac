from openai import APIStatusError

import json
from kino import FilmPick, Recommendation, validate_picks

import httpx
from openai import APIConnectionError

def test_foreign_film_id():
    # Отклоняем ID, которого нет в каталоге
    films = [{"id": "film_001"}]

    result = Recommendation(
        status="ok",
        reason="",
        picks=[
            FilmPick(
                film_id="film_999",
                why="Подходит под запрос",
            )
        ],
    )

    assert validate_picks(result, films) is False

def test_duplicate_film_ids():
    # Один фильм нельзя рекомендовать дважды
    films = [{"id": "film_001"}]

    result = Recommendation(
        status="ok",
        reason="",
        picks=[
            FilmPick(
                film_id="film_001",
                why="Первая рекомендация",
            ),
            FilmPick(
                film_id="film_001",
                why="Повторная рекомендация",
            ),
        ],
    )

    assert validate_picks(result, films) is False

def test_no_match():
    # При отсутствии совпадений список рекомендаций пуст
    films = [{"id": "film_001"}]

    result = Recommendation(
        status="no_match",
        reason="Подходящих фильмов нет",
        picks=[],
    )

    assert validate_picks(result, films) is True

from types import SimpleNamespace

from kino import pick


def test_pick_rejects_foreign_id(tmp_path, monkeypatch):
    # Журнал создаётся во временной папке
    monkeypatch.chdir(tmp_path)

    # Имитируем ответ модели с чужим ID
    result = Recommendation(
        status="ok",
        reason="",
        picks=[
            FilmPick(
                film_id="film_999",
                why="Подходит под запрос",
            )
        ],
    )

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    refusal=None,
                    parsed=result,
                ),
            )
        ]
    )

    class FakeCompletions:
        def parse(self, **kwargs):
            return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(),
        )
    )

    films = [{"id": "film_001"}]

    outcome, ids = pick(
        client,
        "test-model",
        "Проверка чужого ID",
        films,
    )

    assert outcome == "invalid"
    assert ids == []

    # Проверяем запись отклонённого ID в журнал
    with open("calls.jsonl", encoding="utf-8") as file:
        record = json.loads(file.readline())

    assert record["model"] == "test-model"
    assert record["outcome"] == "invalid"
    assert record["rejected"] == ["film_999"]

def test_pick_no_match(tmp_path, monkeypatch):
    # Используем временную папку для журнала
    monkeypatch.chdir(tmp_path)

    result = Recommendation(
        status="no_match",
        reason="Подходящих фильмов нет",
        picks=[],
    )

    # Имитируем корректный ответ модели
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    refusal=None,
                    parsed=result,
                ),
            )
        ]
    )

    class FakeCompletions:
        def parse(self, **kwargs):
            return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(),
        )
    )

    outcome, ids = pick(
        client,
        "test-model",
        "Фильм продолжительностью 999 минут",
        [{"id": "film_001"}],
    )

    assert outcome == "no_match"
    assert ids == []

    # Проверяем результат в журнале
    with open("calls.jsonl", encoding="utf-8") as file:
        record = json.loads(file.readline())

    assert record["outcome"] == "no_match"
    assert record["ids"] == []

def test_pick_refusal(tmp_path, monkeypatch):
    # Записываем журнал во временную папку
    monkeypatch.chdir(tmp_path)

    # Имитируем отказ модели
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    refusal="Не могу выполнить запрос",
                    parsed=None,
                ),
            )
        ]
    )

    class FakeCompletions:
        def parse(self, **kwargs):
            return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(),
        )
    )

    outcome, ids = pick(
        client,
        "test-model",
        "Тест отказа модели",
        [{"id": "film_001"}],
    )

    assert outcome == "refusal"
    assert ids == []

    # Проверяем, что отказ записан в журнал
    with open("calls.jsonl", encoding="utf-8") as file:
        record = json.loads(file.readline())

    assert record["outcome"] == "refusal"
    assert record["ids"] == []

def test_pick_length(tmp_path, monkeypatch):
    # Журнал записываем во временную папку
    monkeypatch.chdir(tmp_path)

    # Имитируем ответ, прерванный лимитом токенов
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(
                    refusal=None,
                    parsed=None,
                ),
            )
        ]
    )

    class FakeCompletions:
        def parse(self, **kwargs):
            return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(),
        )
    )

    outcome, ids = pick(
        client,
        "test-model",
        "Тест ограничения длины ответа",
        [{"id": "film_001"}],
    )

    assert outcome == "length"
    assert ids == []

    # Проверяем запись результата в журнал
    with open("calls.jsonl", encoding="utf-8") as file:
        record = json.loads(file.readline())

    assert record["outcome"] == "length"
    assert record["ids"] == []

def test_pick_no_connection(tmp_path, monkeypatch):
    # Используем временную папку для журнала
    monkeypatch.chdir(tmp_path)

    # Имитируем недоступность Ollama
    class FakeCompletions:
        def parse(self, **kwargs):
            request = httpx.Request(
                "POST",
                "http://localhost:11434/v1/chat/completions",
            )
            raise APIConnectionError(request=request)

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(),
        )
    )

    outcome, ids = pick(
        client,
        "test-model",
        "Тест ошибки подключения",
        [{"id": "film_001"}],
    )

    assert outcome == "no_connection"
    assert ids == []

    # Проверяем запись ошибки в журнал
    with open("calls.jsonl", encoding="utf-8") as file:
        record = json.loads(file.readline())

    assert record["outcome"] == "no_connection"
    assert record["ids"] == []

def test_pick_api_500(tmp_path, monkeypatch):
    # Используем временную папку для журнала
    monkeypatch.chdir(tmp_path)

    # Имитируем ошибку сервера HTTP 500
    class FakeCompletions:
        def parse(self, **kwargs):
            request = httpx.Request(
                "POST",
                "http://localhost:11434/v1/chat/completions",
            )
            response = httpx.Response(
                status_code=500,
                request=request,
            )
            raise APIStatusError(
                "Internal Server Error",
                response=response,
                body=None,
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(),
        )
    )

    outcome, ids = pick(
        client,
        "test-model",
        "Тест ошибки HTTP 500",
        [{"id": "film_001"}],
    )

    assert outcome == "error_500"
    assert ids == []

    # Проверяем запись ошибки в журнал
    with open("calls.jsonl", encoding="utf-8") as file:
        record = json.loads(file.readline())

    assert record["outcome"] == "error_500"
    assert record["ids"] == []