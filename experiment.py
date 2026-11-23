import json

from openai import OpenAI

# Загружаем фильмы для эксперимента
with open("films.json", encoding="utf-8") as file:
    films = json.load(file)[:10]

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# Проверяем знания модели без передачи каталога
for film in films:
    response = client.chat.completions.create(
        model="qwen3.5:4b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Назови год выпуска и продолжительность фильма "
                    "в минутах. Ответь кратко. "
                    "Если не знаешь, так и скажи."
                ),
            },
            {
                "role": "user",
                "content": film["title"],
            },
        ],
        reasoning_effort="none",
    )

    print(f"\nФильм: {film['title']}")
    print(f"Каталог: {film['year']}, {film['minutes']} мин.")
    print(f"Ответ LLM: {response.choices[0].message.content}")