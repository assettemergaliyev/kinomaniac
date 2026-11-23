from collections import Counter
import json



def load_calls() -> list[dict]:
    # Читаем журнал, если он существует
    try:
        with open("calls.jsonl", encoding="utf-8") as file:
            return [
                json.loads(line)
                for line in file
                if line.strip()
            ]
    except FileNotFoundError:
        return []

if __name__ == "__main__":
    calls = load_calls()
    print(f"Всего вызовов: {len(calls)}")
    # Считаем количество вызовов по результатам
    outcomes = Counter(call["outcome"] for call in calls)

    for outcome, count in sorted(outcomes.items()):
        print(f"{outcome}: {count}")

    # Считаем количество отклонённых ID
    rejected = sum(len(call.get("rejected", [])) for call in calls)
    rejection_rate = rejected / len(calls) * 100 if calls else 0

    print(f"Отклонено: {rejected}")
    print(f"Доля отклонений: {rejection_rate:.1f}%")

    # Суммируем токены из записей, где есть usage
    calls_with_usage = [
        call for call in calls
        if isinstance(call.get("usage"), dict)
    ]

    prompt_tokens = sum(
        call["usage"].get("prompt_tokens", 0) or 0
        for call in calls_with_usage
    )
    completion_tokens = sum(
        call["usage"].get("completion_tokens", 0) or 0
        for call in calls_with_usage
    )

    print(f"Вызовов со статистикой токенов: {len(calls_with_usage)}")
    print(f"Входных токенов: {prompt_tokens}")
    print(f"Выходных токенов: {completion_tokens}")
    print(f"Всего токенов: {prompt_tokens + completion_tokens}")

    # Локальная Ollama и тестовая модель не требуют оплаты API
    local_models = {"qwen3.5:4b", "test-model"}
    all_local = all(
        call.get("model") in local_models
        for call in calls
    )

    if all_local:
        print("Стоимость API: $0.0000")
    else:
        print("Стоимость API: не рассчитана (неизвестные тарифы моделей)")
