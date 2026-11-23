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
