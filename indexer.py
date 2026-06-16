"""
indexer.py — Индексация романа "Отцы и дети" с поддержкой русского языка.

Обязанности:
    - Загрузка документов из папки (файлы .txt — главы романа)
    - Автоопределение кодировки файлов (поддерживает cp1251, utf-8 и другие)
    - Обогащение каждой главы заголовком перед разбиением на чанки
    - Разбиение на чанки с помощью RecursiveCharacterTextSplitter
      с разделителями, оптимизированными для русского языка
    - Добавление чанков с метаданными в ChromaDB (в процессе, без сервера)
    - Предоставление интерфейсов поиска и получения статуса

Модель эмбеддингов:
    sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    (оптимизирована для многоязычного текста, включая русский)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import chardet
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательные функции для работы с файлами
# ---------------------------------------------------------------------------

def _read_file_with_encoding(path: Path) -> str:
    """Читает файл с автоматическим определением кодировки через chardet.

    Поддерживает cp1251, utf-8 и другие кодировки, часто встречающиеся
    в русских текстовых файлах. При неудаче использует utf-8 с заменой
    нераспознанных символов.
    """
    raw = path.read_bytes()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        logger.warning(
            "Не удалось декодировать %s как %s, откат на utf-8 с заменой",
            path, encoding
        )
        return raw.decode("utf-8", errors="replace")


def _chunk_id(source: str, chunk_index: int) -> str:
    """Детерминированный идентификатор чанка на основе пути и позиции."""
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Индексатор
# ---------------------------------------------------------------------------

class Indexer:
    """Управляет загрузкой документов и операциями с векторным хранилищем.

    Использует ChromaDB в процессе с мультиязычной моделью Sentence Transformer,
    оптимизированной для русского и других языков.
    """

    def __init__(self) -> None:
        # Функция эмбеддингов для мультиязычного текста (включая русский)
        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg.EMBEDDING_MODEL
        )

        self._client = chromadb.PersistentClient(
            path=cfg.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        self._collection = self._client.get_or_create_collection(
            name=cfg.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_function
        )

        self._last_indexed_at: datetime | None = None
        self._indexed_files: list[str] = []

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def index_folder(self, folder_path: str, glob_pattern: str = "*.txt") -> dict:
        """Загружает все главы (.txt) из папки и индексирует их.

        Каждый файл считается главой. Имя файла (без расширения) используется
        как название главы и добавляется в начало содержимого перед разбиением
        на чанки для улучшения релевантности поиска.

        Аргументы:
            folder_path: Абсолютный или относительный путь к папке с главами.
            glob_pattern: Шаблон для фильтрации файлов (по умолчанию: *.txt).

        Возвращает:
            Словарь с ключами: files_indexed, chunks_added, errors.
        """
        root = Path(folder_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Папка не найдена: {root}")

        # Фильтруем только поддерживаемые расширения из конфига
        files = sorted([
            p for p in root.glob(glob_pattern)
            if p.suffix in cfg.SUPPORTED_EXTENSIONS
        ])
        if not files:
            logger.info("Файлы в корне не найдены, выполняем рекурсивный поиск...")
            files = sorted([
                p for p in root.rglob(glob_pattern)
                if p.suffix in cfg.SUPPORTED_EXTENSIONS
            ])

        logger.info("Найдено файлов для индексации: %d", len(files))

        chunks_added = 0
        errors: list[str] = []

        for file_path in files:
            try:
                new_chunks = self._index_file(file_path)
                chunks_added += new_chunks
                self._indexed_files.append(str(file_path))
                logger.info("  ✓ %s: %d чанков", file_path.name, new_chunks)
            except Exception as exc:
                logger.exception("  ❌ %s: %s", file_path.name, exc)
                errors.append(f"{file_path}: {exc}")

        self._last_indexed_at = datetime.now(timezone.utc)

        return {
            "files_indexed": len(files) - len(errors),
            "chunks_added": chunks_added,
            "errors": errors,
        }

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Возвращает top_k наиболее релевантных чанков для запроса.

        Аргументы:
            query: Поисковый запрос на естественном языке (на русском или английском).
            top_k: Количество результатов (по умолчанию из конфига cfg.TOP_K).

        Возвращает:
            Список словарей с ключами: text, source, chapter, chunk_index,
            distance, char_count.
        """
        k = top_k if top_k is not None else cfg.TOP_K

        # Проверяем, чтобы не запросить больше результатов, чем существует
        count = self._collection.count() or 1
        n_results = min(k, count)

        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[dict] = []
        if not results["documents"] or not results["documents"][0]:
            return chunks

        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": text,
                "source": meta.get("source", "неизвестно"),
                "chapter": meta.get("chapter", "неизвестно"),
                "chunk_index": meta.get("chunk_index", -1),
                "distance": dist,
                "char_count": meta.get("char_count", len(text)),
            })
        return chunks

    def get_status(self) -> dict:
        """Возвращает текущую статистику индекса.

        Возвращает:
            Словарь с ключами: total_chunks, files_count, collection_name,
            last_indexed_at, chroma_db_path.
        """
        return {
            "total_chunks": self._collection.count(),
            "files_count": len(set(self._indexed_files)),
            "collection_name": cfg.COLLECTION_NAME,
            "last_indexed_at": self._last_indexed_at.isoformat() if self._last_indexed_at else None,
            "chroma_db_path": cfg.CHROMA_DB_PATH,
        }

    # ------------------------------------------------------------------
    # Внутренние вспомогательные методы
    # ------------------------------------------------------------------

    def _index_file(self, file_path: Path) -> int:
        """Загружает, обогащает, разбивает на чанки и добавляет один файл главы.

        Возвращает количество добавленных чанков.
        """
        content = _read_file_with_encoding(file_path)
        if not content.strip():
            logger.debug("Пропуск пустого файла: %s", file_path)
            return 0

        # Используем имя файла (без расширения) как название главы
        chapter_name = file_path.stem

        # Добавляем название главы в начало для улучшения контекста при поиске
        enhanced_content = f"Глава: {chapter_name}\n\n{content}"

        # Создаём разделитель, оптимизированный для русского языка
        # Длинные разделители проверяются первыми (абзацы, затем предложения)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.CHUNK_SIZE,
            chunk_overlap=cfg.CHUNK_OVERLAP,
            separators=[
                "\n\n\n",       # Несколько пустых строк (крупные разрывы)
                "\n\n",         # Разрывы абзацев
                "\n",           # Переводы строк
                ". ",           # Концы предложений
                "! ",           # Восклицания
                "? ",           # Вопросы
                "; ",           # Точки с запятой
                ", ",           # Запятые
                " ",            # Границы слов
                ""              # Посимвольное деление (на крайний случай)
            ],
            length_function=len,
            is_separator_regex=False
        )

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "source": str(file_path),
                "chapter": chapter_name,
                "filename": file_path.name
            }
        )

        chunks = splitter.split_documents([doc])
        if not chunks:
            logger.debug("Не создано ни одного чанка из %s", file_path.name)
            return 0

        ids = []
        texts = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(_chunk_id(str(file_path), i))
            texts.append(chunk.page_content)
            metadatas.append({
                "source": str(file_path),
                "chunk_index": i,
                "chapter": chapter_name,
                "char_count": len(chunk.page_content)
            })

        # Добавляем пакетами для производительности
        batch_size = cfg.UPSERT_BATCH_SIZE
        for i in range(0, len(ids), batch_size):
            self._collection.upsert(
                ids=ids[i:i + batch_size],
                documents=texts[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )

        logger.debug("Проиндексировано %d чанков из %s", len(chunks), file_path.name)
        return len(chunks)