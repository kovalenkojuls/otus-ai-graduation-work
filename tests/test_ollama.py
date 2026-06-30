#!/usr/bin/env python3
"""Минимальный тест подключения к Ollama через Python."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
import config as cfg

print("1️⃣ Подключение к Ollama...")
print(f"   URL: {cfg.LLM_BASE_URL}")
print(f"   Модель: {cfg.LLM_MODEL}")

try:
    llm = ChatOpenAI(
        base_url=cfg.LLM_BASE_URL,
        api_key=cfg.LLM_API_KEY,
        model=cfg.LLM_MODEL,
        temperature=0.0,
        timeout=30,
    )

    print("2️⃣ Отправка тестового запроса...")
    response = llm.invoke("Ответь одним словом: OK")
    print(f"   Ответ: {response.content}")
    print("✅ Python -> Ollama работает!")

except Exception as e:
    print(f"❌ Ошибка: {e}")