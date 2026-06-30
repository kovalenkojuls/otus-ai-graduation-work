"""
server.py — FastMCP сервер для RAG-системы по роману "Отцы и дети".

Предоставляет следующие MCP инструменты:
    - index_folder        : Индексирует все главы романа в ChromaDB
    - ask_question        : Запускает полный пайплайн Corrective RAG для вопроса
    - find_relevant_docs  : Ищет релевантные чанки без генерации ответа
    - summarize_document  : Создаёт краткое изложение отдельной главы
    - index_status        : Показывает статистику векторного индекса
    - llm_status          : Проверяет доступность LLM (LM Studio)

Использование:
    python server.py

    Или через конфигурацию MCP клиента:
    {
        "type": "stdio",
        "command": "python",
        "args": ["server.py"],
        "cwd": "<корень проекта>"
    }
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastmcp import FastMCP
from langchain_openai import ChatOpenAI

import config as cfg
from indexer import Indexer, _read_file_with_encoding
from graph import run_graph
from prompts import SUMMARIZATION_PROMPT, strip_thinking_tags


# ---------------------------------------------------------------------------
# Вспомогательные функции для диагностики соединения
# ---------------------------------------------------------------------------

def _is_connection_error(exc: BaseException) -> bool:
    """Определяет, является ли исключение ошибкой сетевого подключения.

    Проверяет само исключение, его причину (__cause__) и контекст (__context__)
    на принадлежность к известным типам ошибок соединения (httpx, httpcore,
    встроенные ошибки Python).

    Аргументы:
        exc: Исключение для проверки

    Возвращает:
        True, если ошибка связана с проблемой сетевого подключения
    """
    import httpx
    import httpcore

    for e in (exc, getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if isinstance(e, (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpcore.ConnectError,
            httpcore.ConnectTimeout,
            ConnectionRefusedError,
            ConnectionError,
        )):
            return True
    return False


def _llm_connection_error_message() -> str:
    """Формирует понятное сообщение об ошибке подключения к LM Studio.

    Возвращает:
        Строку с инструкциями по запуску LM Studio
    """
    return (
        f"Не удалось подключиться к LM Studio по адресу {cfg.LLM_BASE_URL}. "
        "Пожалуйста, убедитесь, что LM Studio запущена на хост-машине, "
        "модель загружена и локальный сервер запущен "
        "(LM Studio → вкладка Local Server → Start Server). "
        f"Ожидаемая модель: '{cfg.LLM_MODEL}'."
    )


# ---------------------------------------------------------------------------
# Настройка логирования
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Синглтоны
# ---------------------------------------------------------------------------
_indexer: Indexer | None = None
_llm: ChatOpenAI | None = None


def _get_indexer() -> Indexer:
    """Возвращает синглтон индексатора, создавая его при первом обращении."""
    global _indexer
    if _indexer is None:
        _indexer = Indexer()
    return _indexer


def _get_llm() -> ChatOpenAI:
    """Возвращает синглтон LLM-клиента, создавая его при первом обращении.

    Использует настройки из config.py для подключения к LM Studio.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            base_url=cfg.LLM_BASE_URL,
            api_key=cfg.LLM_API_KEY,
            model=cfg.LLM_MODEL,
            temperature=cfg.LLM_TEMPERATURE,
        )
    return _llm


# ---------------------------------------------------------------------------
# Создание FastMCP сервера
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="otsi-i-deti-rag-server",
    instructions=(
        "RAG-сервер для романа И.С. Тургенева «Отцы и дети». "
        "ВСЕГДА сначала вызывай index_status, чтобы проверить, проиндексированы ли уже главы. "
        "Если total_chunks равен 0, вызови index_folder с путём к папке с главами романа "
        "перед тем, как задавать вопросы. "
        "Если ask_question возвращает ошибку подключения, вызови llm_status для диагностики, "
        "запущена ли LM Studio и загружена ли модель. "
        "Используй ask_question для запросов к тексту романа через полный пайплайн Corrective RAG. "
        "Используй find_relevant_docs для поиска чанков без генерации ответа (отладка). "
        "Используй summarize_document для краткого изложения отдельной главы. "
        "Используй index_status для проверки количества проиндексированных чанков "
        "и времени последней индексации. "
        "Используй llm_status для проверки доступности LLM-бэкенда."
    ),
)


# ---------------------------------------------------------------------------
# Инструмент 1 — index_folder
# ---------------------------------------------------------------------------

@mcp.tool()
def index_folder(folder_path: str, glob_pattern: str = "*.txt") -> dict:
    """Индексирует все главы романа (.txt) из указанной папки в векторное хранилище.

    Каждый .txt файл считается отдельной главой. Имя файла (без расширения)
    используется как название главы и добавляется в метаданные чанков
    для улучшения релевантности поиска и цитирования.

    Поддерживаемые форматы: .txt (с автоопределением кодировки, включая cp1251)

    Аргументы:
        folder_path: Путь к папке с главами романа (например, "./chapters")
        glob_pattern: Шаблон имён файлов (по умолчанию: "*.txt")

    Возвращает:
        Словарь с ключами:
            - files_indexed (int): Количество успешно проиндексированных файлов
            - chunks_added (int): Общее количество добавленных чанков
            - errors (list[str]): Список ошибок (пустой, если всё прошло успешно)

    Пример:
        index_folder("./chapters")
        → {"files_indexed": 28, "chunks_added": 456, "errors": []}
    """
    logger.info("index_folder вызван: path=%s glob=%s", folder_path, glob_pattern)
    try:
        result = _get_indexer().index_folder(folder_path, glob_pattern)
        logger.info(
            "Проиндексировано файлов: %d, чанков: %d, ошибок: %d",
            result["files_indexed"],
            result["chunks_added"],
            len(result["errors"]),
        )
        return result
    except FileNotFoundError as exc:
        logger.error("Папка не найдена: %s", exc)
        return {
            "files_indexed": 0,
            "chunks_added": 0,
            "errors": [f"Папка не найдена: {exc}"],
        }
    except Exception as exc:
        logger.exception("Неожиданная ошибка в index_folder")
        return {
            "files_indexed": 0,
            "chunks_added": 0,
            "errors": [f"Неожиданная ошибка: {exc}"],
        }


# ---------------------------------------------------------------------------
# Инструмент 2 — ask_question
# ---------------------------------------------------------------------------

@mcp.tool()
def ask_question(question: str) -> dict:
    """Задаёт вопрос по тексту романа «Отцы и дети».

    Запускает полный пайплайн Corrective RAG:
    1. Переписывание вопроса для улучшения поиска
    2. Поиск релевантных чанков в ChromaDB
    3. Оценка релевантности найденных чанков
    4. Генерация ответа на основе контекста
    5. Проверка ответа на галлюцинации (с автоматическим повтором при необходимости)

    Аргументы:
        question: Вопрос на естественном языке (русский или английский)

    Возвращает:
        Словарь с ключами:
            - answer (str): Сгенерированный ответ с цитированием источников
            - sources (list[str]): Список файлов глав, использованных в ответе
            - is_grounded (bool): Прошёл ли ответ проверку на фактическую обоснованность
            - retrieve_retries (int): Количество повторных попыток поиска
            - generate_retries (int): Количество повторных попыток генерации
            - relevant_chunks (int): Количество найденных релевантных чанков

    Пример:
        ask_question("Почему Базаров и Павел Петрович стрелялись на дуэли?")
        → {
            "answer": "Дуэль произошла из-за того, что Павел Петрович увидел...",
            "sources": ["chapter_24.txt"],
            "is_grounded": true,
            ...
        }
    """
    logger.info("ask_question вызван: %s", question[:120])

    # Проверяем, есть ли проиндексированные данные
    if _get_indexer().get_status()["total_chunks"] == 0:
        return {
            "answer": (
                "Индекс пуст. Сначала необходимо проиндексировать главы романа. "
                "Вызовите index_folder с путём к папке, содержащей .txt файлы глав "
                "(например, index_folder('./chapters'))."
            ),
            "sources": [],
            "is_grounded": False,
            "retrieve_retries": 0,
            "generate_retries": 0,
            "relevant_chunks": 0,
        }

    try:
        result = run_graph(question, indexer=_get_indexer())
        return {
            "answer": result["generation"],
            "sources": result["sources"],
            "is_grounded": result["is_grounded"],
            "retrieve_retries": result["retrieve_retry_count"],
            "generate_retries": result["generate_retry_count"],
            "relevant_chunks": result["relevant_chunks_count"],
        }
    except Exception as exc:
        logger.exception("Ошибка в ask_question")

        if _is_connection_error(exc):
            answer = _llm_connection_error_message()
        else:
            answer = f"Ошибка выполнения RAG пайплайна: {exc}"

        return {
            "answer": answer,
            "sources": [],
            "is_grounded": False,
            "retrieve_retries": 0,
            "generate_retries": 0,
            "relevant_chunks": 0,
        }


# ---------------------------------------------------------------------------
# Инструмент 3 — find_relevant_docs
# ---------------------------------------------------------------------------

@mcp.tool()
def find_relevant_docs(query: str, top_k: int = 5) -> dict:
    """Ищет наиболее релевантные чанки по запросу без генерации ответа.

    Полезно для:
    - Проверки качества индексации
    - Отладки поисковой выдачи
    - Быстрого поиска фрагментов текста без запуска полного пайплайна

    Аргументы:
        query: Поисковый запрос (например, "описание дуэли Базарова")
        top_k: Максимальное количество возвращаемых чанков (по умолчанию: 5)

    Возвращает:
        Словарь с ключами:
            - results (list[dict]): Список чанков, каждый с полями:
                - text (str): Текст чанка
                - source (str): Путь к файлу главы
                - chapter (str): Название главы
                - chunk_index (int): Индекс чанка в главе
                - distance (float): Косинусное расстояние (меньше = более релевантный)
                - char_count (int): Количество символов в чанке
            - count (int): Общее количество найденных чанков
            - message (str): Сообщение, если индекс пуст

    Пример:
        find_relevant_docs("смерть Базарова", top_k=3)
        → {"results": [...], "count": 3}
    """
    logger.info("find_relevant_docs вызван: query=%s top_k=%d", query[:80], top_k)

    if _get_indexer().get_status()["total_chunks"] == 0:
        return {
            "results": [],
            "count": 0,
            "message": "Индекс пуст. Сначала выполните index_folder.",
        }

    try:
        chunks = _get_indexer().retrieve(query, top_k=top_k)
        return {
            "results": chunks,
            "count": len(chunks),
        }
    except Exception as exc:
        logger.exception("Ошибка в find_relevant_docs")
        return {
            "results": [],
            "count": 0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Инструмент 4 — summarize_document
# ---------------------------------------------------------------------------

@mcp.tool()
def summarize_document(file_path: str) -> dict:
    """Создаёт краткое изложение отдельной главы романа с помощью LLM.

    Файл читается напрямую (не требует предварительной индексации)
    и передаётся в LLM с шаблоном для суммаризации.

    Для очень больших файлов текст автоматически обрезается до безопасного
    размера, чтобы избежать переполнения контекстного окна модели.

    Аргументы:
        file_path: Путь к файлу главы (например, "./chapters/chapter_24.txt")

    Возвращает:
        Словарь с ключами:
            - summary (str): Структурированное краткое изложение главы, включающее:
                - Название файла и главы
                - Краткое содержание (2-4 предложения)
                - Основные темы (список)
                - Персонажи (список)
            - filename (str): Имя файла главы
            - error (str): Сообщение об ошибке (если есть)

    Пример:
        summarize_document("./chapters/chapter_24.txt")
        → {
            "summary": "**Файл:** chapter_24.txt\\n**Глава:** chapter_24\\n...",
            "filename": "chapter_24.txt"
        }
    """
    logger.info("summarize_document вызван: %s", file_path)
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        logger.error("Файл не найден: %s", path)
        return {
            "summary": "",
            "filename": path.name,
            "error": f"Файл не найден: {path}",
        }

    try:
        content = _read_file_with_encoding(path)

        if not content.strip():
            return {
                "summary": "Файл пуст.",
                "filename": path.name,
            }

        # Ограничиваем размер для предотвращения переполнения контекстного окна
        max_chars = cfg.CHUNK_SIZE * 10  # примерно 15 000 символов
        if len(content) > max_chars:
            logger.warning(
                "Файл %s слишком большой (%d символов), обрезаем до %d",
                path.name, len(content), max_chars
            )
            content = content[:max_chars] + "\n\n[... текст обрезан для суммаризации ...]"

        # Определяем название главы из имени файла
        chapter_name = path.stem

        chain = SUMMARIZATION_PROMPT | _get_llm()
        result = chain.invoke({
            "filename": path.name,
            "chapter": chapter_name,
            "document": content,
        })

        raw = result.content if hasattr(result, "content") else str(result)
        if cfg.LLM_STRIP_THINKING_TAGS:
            raw = strip_thinking_tags(raw)

        logger.info("Суммаризация завершена для %s", path.name)
        return {
            "summary": raw.strip(),
            "filename": path.name,
        }

    except Exception as exc:
        logger.exception("Ошибка в summarize_document")
        if _is_connection_error(exc):
            return {
                "summary": "",
                "filename": str(path),
                "error": _llm_connection_error_message(),
            }
        return {
            "summary": "",
            "filename": str(path),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Инструмент 5 — index_status
# ---------------------------------------------------------------------------

@mcp.tool()
def index_status() -> dict:
    """Показывает статистику текущего состояния векторного индекса.

    Возвращает информацию о количестве проиндексированных чанков,
    файлов, времени последней индексации и расположении хранилища.

    Вызывайте этот инструмент перед ask_question, чтобы убедиться,
    что главы романа уже проиндексированы.

    Возвращает:
        Словарь с ключами:
            - total_chunks (int): Общее количество чанков в индексе
            - files_count (int): Количество уникальных проиндексированных файлов
            - collection_name (str): Название коллекции в ChromaDB
            - last_indexed_at (str | None): ISO-время последней индексации (или None)
            - chroma_db_path (str): Путь к папке с файлами ChromaDB

    Пример:
        index_status()
        → {
            "total_chunks": 456,
            "files_count": 28,
            "collection_name": "fathers_and_sons_ru",
            "last_indexed_at": "2024-01-15T12:30:00+00:00",
            "chroma_db_path": "./book_chroma_db_ru"
        }
    """
    logger.info("index_status вызван")
    return _get_indexer().get_status()


# ---------------------------------------------------------------------------
# Инструмент 6 — llm_status
# ---------------------------------------------------------------------------

@mcp.tool()
def llm_status() -> dict:
    """Проверяет доступность LLM-бэкенда (LM Studio).

    Отправляет минимальный тестовый запрос к настроенному эндпоинту LLM.
    Используйте для диагностики проблем подключения перед вызовом ask_question
    или summarize_document.

    Возвращает:
        Словарь с ключами:
            - ok (bool): True, если LLM ответил успешно
            - model (str): Название настроенной модели
            - base_url (str): URL эндпоинта LLM
            - error (str | None): Сообщение об ошибке, если LLM недоступен

    Пример (успех):
        llm_status()
        → {"ok": true, "model": "qwen2.5-7b-instruct-1m", "base_url": "http://localhost:1234/v1", "error": null}

    Пример (ошибка):
        llm_status()
        → {"ok": false, "model": "...", "base_url": "...", "error": "Не удалось подключиться к LM Studio..."}
    """
    logger.info("llm_status вызван")
    try:
        llm = _get_llm()
        resp = llm.invoke("Ответь одним словом: ОК")
        logger.info("LLM доступен: %s", cfg.LLM_MODEL)
        return {
            "ok": True,
            "model": cfg.LLM_MODEL,
            "base_url": cfg.LLM_BASE_URL,
            "error": None,
        }
    except Exception as exc:
        if _is_connection_error(exc):
            msg = _llm_connection_error_message()
        else:
            msg = str(exc)
        logger.warning("llm_status: LLM недоступен: %s", msg)
        return {
            "ok": False,
            "model": cfg.LLM_MODEL,
            "base_url": cfg.LLM_BASE_URL,
            "error": msg,
        }


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    """Запускает MCP-сервер.

    Транспорт выбирается через переменную окружения TRANSPORT:
      - ``stdio``             (по умолчанию) — для интеграции с MCP-клиентами (Claude Desktop)
      - ``streamable-http``  — HTTP-сервер на MCP_HOST:MCP_PORT, используется в Docker
      - ``sse``              — устаревший SSE-транспорт

    Переменные окружения:
      TRANSPORT   stdio | streamable-http | sse  (по умолчанию: stdio)
      MCP_HOST    адрес для HTTP-транспорта       (по умолчанию: 127.0.0.1)
      MCP_PORT    порт для HTTP-транспорта         (по умолчанию: 8000)
    """
    import os

    transport = os.environ.get("TRANSPORT", "stdio")
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT", "8000"))

    logger.info(
        "Запуск MCP RAG-сервера «Отцы и дети» (модель: %s, транспорт: %s)",
        cfg.LLM_MODEL, transport,
    )

    # Выводим сводку о состоянии индекса при запуске
    status = _get_indexer().get_status()
    if status["total_chunks"] > 0:
        logger.info(
            "Индекс уже содержит %d чанков из %d файлов. "
            "Последняя индексация: %s",
            status["total_chunks"],
            status["files_count"],
            status["last_indexed_at"] or "неизвестно",
        )
    else:
        logger.info(
            "Индекс пуст. Используйте инструмент index_folder для индексации глав романа."
        )

    # Запускаем сервер с выбранным транспортом
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport in ("streamable-http", "sse"):
        mcp.run(transport=transport, host=host, port=port)
    else:
        raise ValueError(
            f"Неизвестный TRANSPORT={transport!r}. "
            "Используйте 'stdio', 'streamable-http' или 'sse'."
        )


if __name__ == "__main__":
    main()