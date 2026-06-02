import re
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class BookSearcher:
    """Поисковик"""

    def __init__(self):
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        self._client = chromadb.PersistentClient(
            path="./book_chroma_db_ru",
            settings=Settings(anonymized_telemetry=False)
        )

        self._collection = self._client.get_collection(
            name="fathers_and_sons_ru",
            embedding_function=self.embedding_function
        )

        print(f"✅ Загружено чанков: {self._collection.count()}")

    def search(self, query: str, n=3):
        """Ищет и возвращает результаты"""

        # Для multilingual-mpnet НЕ нужен префикс "query: " (он для E5)
        results = self._collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )

        shown = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            relevance = round(max(0, 1 - dist), 2)

            # Убираем префикс "Глава: ..."
            if "\n\n" in text:
                text_clean = text.split("\n\n", 1)[1]
            else:
                text_clean = text

            shown.append({
                "chapter": meta.get("chapter", "").replace("_", " "),
                "relevance": relevance,
                "text": text_clean[:300].strip()
            })

        return shown


def test():
    """Тест"""

    print("=" * 50)
    print("ПОИСК ПО 'ОТЦЫ И ДЕТИ'")
    print("=" * 50)

    searcher = BookSearcher()

    queries = [
        "Базаров и Одинцова любовь",
        "дуэль Базарова и Павла",
        "смерть Базарова",
        "родители Базарова",
        "природа не храм а мастерская",
    ]

    for q in queries:
        print(f"\n🔍 «{q}»")
        print("-" * 40)

        results = searcher.search(q)

        if not results:
            print("  Ничего не найдено")
            continue

        for r in results[:3]:
            print(f"  [{r['chapter']}] совпадение: {r['relevance']}")
            print(f"  {r['text']}\n")


if __name__ == "__main__":
    test()