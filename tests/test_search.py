"""
test_search.py — Быстрая проверка поиска по роману "Отцы и дети".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer import Indexer


def test():
    """Основная функция тестирования."""

    print("=" * 60)
    print("  ТЕСТИРОВАНИЕ ПОИСКА ПО 'ОТЦЫ И ДЕТИ'")
    print("=" * 60)

    indexer = Indexer()

    status = indexer.get_status()
    print(f"\n📊 Статус: {status['total_chunks']} чанков, {status['files_count']} файлов")

    queries = [
        "Базаров и Одинцова",
        "нигилизм и отрицание авторитетов",
        "дуэль Павла Петровича",
        "смерть Базарова",
        "природа не храм а мастерская",
    ]

    print("\n" + "-" * 60)

    for query in queries:
        print(f"\n🔍 «{query}»")
        print("-" * 40)

        results = indexer.retrieve(query, top_k=2)

        if not results:
            print("  ❌ Ничего не найдено")
            continue

        for i, r in enumerate(results):
            relevance = max(0, 1 - r['distance'])
            text = r['text'].replace('\n', ' ')[:200]

            print(f"  {i+1}. [{r['chapter']}] релевантность: {relevance:.2f}")
            print(f"     {text}...")

    print("\n" + "=" * 60)
    print("  ✅ Тестирование завершено!")
    print("=" * 60)


if __name__ == "__main__":
    test()