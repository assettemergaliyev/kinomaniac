from collections import Counter
from openai import OpenAI
from kino import pick
import json


def load_queries() -> list[dict]:
    # Загружаем тестовые запросы
    with open("queries.json", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    queries = load_queries()

    # Загружаем каталог фильмов
    with open("films.json", encoding="utf-8") as file:
        films = json.load(file)

    # Подключаемся к локальной Ollama
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    # Считаем количество каждого исхода
    outcomes = Counter()
    # Проверяем, насколько рекомендации совпадают с ожиданиями
    must_total = 0
    must_passed = 0
    # Запоминаем количество записей до проверки
    with open("calls.jsonl", encoding="utf-8") as file:
        calls_before = sum(1 for line in file if line.strip())

    # Последовательно проверяем все запросы
    for number, item in enumerate(queries, start=1):
        query = item["query"]

        print(f"\n{'=' * 50}")
        print(f"Запрос {number}/{len(queries)}: {query}")

        outcome, ids = pick(client, "qwen3.5:4b", query, films)
        outcomes[outcome] += 1

        expected_ids = item.get("expect", {}).get("must", [])

        if expected_ids:
            must_total += 1

            if outcome == "ok" and any(
                film_id in ids for film_id in expected_ids
            ):
                must_passed += 1

        print(f"Результат: {outcome}")
        print(f"ID фильмов: {ids}")

    # Итоговая статистика по запросам
    print("\nИтоги проверки:")
    print(f"Всего запросов: {len(queries)}")

    for status in ("ok", "no_match", "invalid"):
        print(f"{status}: {outcomes[status]}")

    # Считаем отклонённые ID только за текущий прогон
    with open("calls.jsonl", encoding="utf-8") as file:
        new_calls = [
            json.loads(line)
            for index, line in enumerate(file)
            if index >= calls_before and line.strip()
        ]

    rejected_count = sum(
        len(call.get("rejected") or [])
        for call in new_calls
    )

    print(f"Отклонено чужих ID: {rejected_count}")
    print(f"Проверок must пройдено: {must_passed}/{must_total}")