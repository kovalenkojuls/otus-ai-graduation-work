"""
graph.py — LangGraph пайплайн Corrective RAG для романа "Отцы и дети".

Поток графа:
    Запрос пользователя
        │
        ▼
    rewrite_query       — улучшить запрос для векторного поиска
        │
        ▼
    retrieve            — получить top-k чанков из ChromaDB
        │
        ▼
    grade_chunks        — отфильтровать нерелевантные чанки
        │
    ┌───┴──────────────────────────────────────────┐
    │ достаточно релевантных                       │ слишком мало релевантных
    ▼                                              ▼
  generate             ←─────────────   rewrite_query (повтор, макс. 2)
    │
    ▼
  hallucination_check
    │
  ┌─┴───────────────────┐
  │ обоснован           │ не обоснован (макс. 1 повтор → generate)
  ▼                     ▼
  END                 generate (повтор)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

import config as cfg
from indexer import Indexer
from prompts import (
    CHUNK_GRADING_PROMPT,
    GENERATION_PROMPT,
    HALLUCINATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    strip_thinking_tags,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Синглтоны уровня модуля (ленивая инициализация при первом вызове run_graph)
# ---------------------------------------------------------------------------
_indexer: Indexer | None = None
_llm: ChatOpenAI | None = None


def _get_indexer() -> Indexer:
    """Возвращает синглтон индексатора, создавая его при необходимости."""
    global _indexer
    if _indexer is None:
        _indexer = Indexer()
    return _indexer


def _get_llm() -> ChatOpenAI:
    """Возвращает синглтон LLM, создавая его при необходимости.

    Использует настройки подключения из config.py.
    По умолчанию ожидает LM Studio на localhost:1234.
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
# Состояние графа
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    """Состояние, передаваемое между узлами графа Corrective RAG.

    Атрибуты:
        question: Исходный вопрос пользователя
        rewritten_query: Улучшенный запрос для векторного поиска
        documents: Сырые чанки, полученные из ChromaDB
        graded_documents: Только релевантные чанки после оценки
        context: Отформатированный контекст для генерации ответа
        generation: Сгенерированный LLM ответ
        sources: Уникальные пути к файлам-источникам
        retrieve_retry_count: Счётчик повторных поисков
        generate_retry_count: Счётчик повторных генераций
        is_grounded: Результат проверки на галлюцинации
    """
    question: str
    rewritten_query: str
    documents: list[dict]
    graded_documents: list[dict]
    context: str
    generation: str
    sources: list[str]
    retrieve_retry_count: int
    generate_retry_count: int
    is_grounded: bool


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _invoke_llm(prompt_template, **kwargs) -> str:
    """Вызывает LLM с шаблоном запроса и возвращает очищенный строковый вывод.

    Аргументы:
        prompt_template: Шаблон ChatPromptTemplate для форматирования
        **kwargs: Переменные для подстановки в шаблон

    Возвращает:
        Очищенную строку ответа от LLM (с удалёнными тегами <think>, если нужно)
    """
    chain = prompt_template | _get_llm()
    result = chain.invoke(kwargs)
    raw = result.content if hasattr(result, "content") else str(result)
    if cfg.LLM_STRIP_THINKING_TAGS:
        raw = strip_thinking_tags(raw)
    return raw.strip()


def _parse_binary_json(text: str, key: str, fallback: str = "yes") -> str:
    """Извлекает значение yes/no из JSON-ответа LLM.

    Пытается распарсить строгий JSON, затем ищет регулярным выражением,
    затем возвращает значение по умолчанию. Это гарантирует, что пайплайн
    не упадёт из-за некорректного вывода модели.

    Аргументы:
        text: Текстовый ответ от LLM
        key: Ключ JSON для извлечения ("relevant" или "grounded")
        fallback: Значение по умолчанию, если парсинг не удался

    Возвращает:
        "yes" или "no" в нижнем регистре
    """
    # Пробуем строгий парсинг JSON
    try:
        # Иногда модель оборачивает JSON в ```json ... ``` fences
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        data = json.loads(cleaned)
        return str(data.get(key, fallback)).lower()
    except (json.JSONDecodeError, AttributeError):
        pass

    # Запасной вариант с регулярным выражением — ищем ключ со значением yes/no
    match = re.search(rf'"{key}"\s*:\s*"(yes|no)"', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Последняя надежда: ищем просто yes/no в любом месте ответа
    lower = text.lower()
    if "yes" in lower:
        return "yes"
    if "no" in lower:
        return "no"

    logger.warning(
        "Не удалось распарсить бинарный JSON для ключа '%s' из: %s — "
        "использую значение по умолчанию '%s'",
        key, text[:200], fallback
    )
    return fallback


def _format_context(chunks: list[dict]) -> tuple[str, list[str]]:
    """Форматирует оценённые чанки в строку контекста и список источников.

    Каждый чанк предваряется информацией об источнике и главе для
    улучшения цитируемости в финальном ответе.

    Аргументы:
        chunks: Список словарей с ключами text, source, chapter, chunk_index

    Возвращает:
        Кортеж из (строка_контекста, список_уникальных_источников)
    """
    parts: list[str] = []
    sources: list[str] = []

    for chunk in chunks:
        source = chunk.get("source", "неизвестно")
        chapter = chunk.get("chapter", "неизвестно")

        # Форматируем каждый чанк с метаданными для улучшения контекста
        parts.append(
            f"[Источник: {source} | Глава: {chapter}]\n{chunk['text']}"
        )

        if source not in sources:
            sources.append(source)

    return "\n\n---\n\n".join(parts), sources


# ---------------------------------------------------------------------------
# Узлы графа
# ---------------------------------------------------------------------------

def rewrite_query(state: RAGState) -> RAGState:
    """Переписывает вопрос пользователя в оптимизированный поисковый запрос.

    Этот узел берёт исходный вопрос и преобразует его в более эффективную
    форму для векторного поиска: убирает разговорные обороты, оставляет
    ключевые слова, имена персонажей и важные термины.

    Пример:
        Вход: "Что ты можешь рассказать про отношения Базарова с родителями?"
        Выход: "отношения Базарова с родителями визит домой старики Базаровы"
    """
    logger.debug("Узел: rewrite_query | попытка=%d", state.get("retrieve_retry_count", 0))
    rewritten = _invoke_llm(QUERY_REWRITE_PROMPT, question=state["question"])
    return {**state, "rewritten_query": rewritten}


def retrieve(state: RAGState) -> RAGState:
    """Извлекает top-k чанков из ChromaDB по переписанному запросу.

    Использует мультиязычную модель эмбеддингов для поиска семантически
    близких фрагментов текста. Возвращает чанки, отсортированные по
    релевантности (косинусное расстояние).
    """
    logger.debug("Узел: retrieve | запрос=%s", state["rewritten_query"])
    docs = _get_indexer().retrieve(state["rewritten_query"])
    logger.debug("Получено чанков: %d", len(docs))
    return {**state, "documents": docs}


def grade_chunks(state: RAGState) -> RAGState:
    """Оценивает каждый полученный чанк на релевантность вопросу.

    Для каждого чанка вызывается LLM с запросом CHUNK_GRADING_PROMPT,
    который возвращает JSON с оценкой "yes" или "no". Оставляются только
    чанки с оценкой "yes". Это ключевой этап фильтрации, который
    предотвращает попадание нерелевантного контекста в генерацию.

    Если релевантных чанков не найдено, состояние остаётся без изменений,
    а маршрутизатор decide_after_grading решит, повторить поиск или нет.
    """
    logger.debug("Узел: grade_chunks | чанков=%d", len(state["documents"]))
    relevant: list[dict] = []

    for i, chunk in enumerate(state["documents"]):
        response = _invoke_llm(
            CHUNK_GRADING_PROMPT,
            question=state["question"],
            document=chunk["text"],
        )
        verdict = _parse_binary_json(response, key="relevant", fallback="yes")

        if verdict == "yes":
            relevant.append(chunk)
            logger.debug("  Чанк %d: релевантен (глава: %s)", i, chunk.get("chapter", "?"))
        else:
            logger.debug("  Чанк %d: НЕ релевантен", i)

    # Форматируем контекст и собираем источники
    context, sources = _format_context(relevant)
    logger.debug("После оценки: %d релевантных чанков из %d", len(relevant), len(state["documents"]))

    return {
        **state,
        "graded_documents": relevant,
        "context": context,
        "sources": sources
    }


def generate(state: RAGState) -> RAGState:
    """Генерирует ответ на основе отфильтрованных релевантных чанков.

    Использует шаблон GENERATION_PROMPT, который требует от модели:
    - Использовать только информацию из контекста
    - Цитировать источники с указанием главы
    - Отвечать на языке вопроса
    - Не добавлять собственные знания

    Если контекст пуст (не найдено релевантных чанков), модель скажет,
    что информации недостаточно.
    """
    logger.debug("Узел: generate | попытка=%d | контекст длиной %d символов",
                 state.get("generate_retry_count", 0),
                 len(state.get("context", "")))

    answer = _invoke_llm(
        GENERATION_PROMPT,
        question=state["question"],
        context=state["context"],
    )
    logger.debug("Сгенерирован ответ длиной %d символов", len(answer))

    return {**state, "generation": answer}


def hallucination_check(state: RAGState) -> RAGState:
    """Проверяет сгенерированный ответ на фактическую обоснованность.

    Вызывает LLM с шаблоном HALLUCINATION_PROMPT, который сравнивает
    каждое утверждение в ответе с контекстом. Возвращает JSON с полем
    "grounded": "yes" или "no".

    Если модель находит хотя бы одно утверждение, не подтверждённое
    контекстом, ответ считается необоснованным и запускается повторная
    генерация (если не исчерпан лимит попыток).
    """
    logger.debug("Узел: hallucination_check")
    response = _invoke_llm(
        HALLUCINATION_PROMPT,
        generation=state["generation"],
        context=state["context"],
    )
    verdict = _parse_binary_json(response, key="grounded", fallback="yes")
    is_grounded = verdict == "yes"

    logger.debug("Результат проверки: %s", "обоснован" if is_grounded else "НЕ обоснован")

    return {**state, "is_grounded": is_grounded}


# ---------------------------------------------------------------------------
# Функции условных переходов
# ---------------------------------------------------------------------------

def _route_after_grading(state: RAGState) -> str:
    """Определяет следующий шаг после оценки чанков.

    Логика:
    - Если есть хотя бы один релевантный чанк → переходим к генерации
    - Если релевантных нет, но есть попытки повтора → переписываем запрос заново
    - Если релевантных нет и попытки исчерпаны → всё равно генерируем
      (best-effort ответ с сообщением о недостатке информации)
    """
    relevant_count = len(state.get("graded_documents", []))
    retry_count = state.get("retrieve_retry_count", 0)

    if relevant_count > 0:
        logger.debug("Маршрутизация: найдено %d релевантных чанков → генерация", relevant_count)
        return "generate"

    if retry_count < cfg.MAX_RETRIEVE_RETRIES:
        logger.debug(
            "Маршрутизация: нет релевантных чанков → повтор поиска (%d/%d)",
            retry_count + 1, cfg.MAX_RETRIEVE_RETRIES
        )
        return "rewrite_query_retry"

    logger.warning(
        "Маршрутизация: нет релевантных чанков после %d попыток → "
        "генерация best-effort ответа", retry_count
    )
    return "generate"


def _route_after_hallucination_check(state: RAGState) -> str:
    """Определяет следующий шаг после проверки на галлюцинации.

    Логика:
    - Если ответ обоснован → завершаем работу (END)
    - Если ответ не обоснован и есть попытки → перегенерируем
    - Если ответ не обоснован и попытки исчерпаны → возвращаем как есть
      (с предупреждением в логах)
    """
    if state.get("is_grounded", True):
        logger.debug("Маршрутизация: ответ обоснован → завершение")
        return END

    retry_count = state.get("generate_retry_count", 0)
    if retry_count < cfg.MAX_GENERATE_RETRIES:
        logger.debug(
            "Маршрутизация: обнаружена галлюцинация → перегенерация (%d/%d)",
            retry_count + 1, cfg.MAX_GENERATE_RETRIES
        )
        return "generate_retry"

    logger.warning(
        "Маршрутизация: ответ всё ещё не обоснован после %d попыток → "
        "возвращаем best-effort ответ", retry_count
    )
    return END


# ---------------------------------------------------------------------------
# Узлы увеличения счётчиков повторов
# ---------------------------------------------------------------------------

def _increment_retrieve_retry(state: RAGState) -> RAGState:
    """Увеличивает счётчик повторных попыток поиска на 1.

    Этот узел — тонкая прослойка перед возвратом к rewrite_query.
    Он гарантирует, что мы не зациклимся бесконечно.
    """
    new_count = state.get("retrieve_retry_count", 0) + 1
    logger.debug("Счётчик повторов поиска увеличен до %d", new_count)
    return {**state, "retrieve_retry_count": new_count}


def _increment_generate_retry(state: RAGState) -> RAGState:
    """Увеличивает счётчик повторных генераций на 1.

    Этот узел — тонкая прослойка перед возвратом к generate.
    Он гарантирует, что мы не зациклимся бесконечно.
    """
    new_count = state.get("generate_retry_count", 0) + 1
    logger.debug("Счётчик повторов генерации увеличен до %d", new_count)
    return {**state, "generate_retry_count": new_count}


# ---------------------------------------------------------------------------
# Построение графа
# ---------------------------------------------------------------------------

def build_graph() -> "CompiledGraph":
    """Строит и компилирует граф Corrective RAG на LangGraph.

    Структура графа:

    1. rewrite_query → retrieve → grade_chunks
       (базовая цепочка: переписать → найти → оценить)

    2. После grade_chunks:
       - Если есть релевантные → generate
       - Если нет → rewrite_query_retry → rewrite_query (цикл до 2 раз)

    3. После generate → hallucination_check
       (всегда проверяем ответ)

    4. После hallucination_check:
       - Если обоснован → END
       - Если нет → generate_retry → generate (цикл до 1 раза)

    Возвращает:
        Скомпилированный граф LangGraph, готовый к выполнению.
    """
    builder = StateGraph(RAGState)

    # Основные узлы
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_chunks", grade_chunks)
    builder.add_node("generate", generate)
    builder.add_node("hallucination_check", hallucination_check)

    # Вспомогательные узлы для повторов
    builder.add_node("rewrite_query_retry", _increment_retrieve_retry)
    builder.add_node("generate_retry", _increment_generate_retry)

    # Точка входа
    builder.set_entry_point("rewrite_query")

    # Линейные рёбра
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "grade_chunks")
    builder.add_edge("generate", "hallucination_check")

    # Рёбра циклов повтора
    builder.add_edge("rewrite_query_retry", "rewrite_query")
    builder.add_edge("generate_retry", "generate")

    # Условные переходы: после оценки чанков
    builder.add_conditional_edges(
        "grade_chunks",
        _route_after_grading,
        {
            "generate": "generate",
            "rewrite_query_retry": "rewrite_query_retry",
        },
    )

    # Условные переходы: после проверки на галлюцинации
    builder.add_conditional_edges(
        "hallucination_check",
        _route_after_hallucination_check,
        {
            END: END,
            "generate_retry": "generate_retry",
        },
    )

    return builder.compile()


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

# Скомпилированный граф — синглтон
_graph = None


def run_graph(query: str, indexer: Indexer | None = None) -> dict:
    """Запускает полный пайплайн Corrective RAG для заданного вопроса.

    Это основная точка входа для внешнего кода. Принимает вопрос
    пользователя, прогоняет его через все этапы графа и возвращает
    структурированный результат.

    Аргументы:
        query: Вопрос пользователя на естественном языке
        indexer: Опциональный экземпляр Indexer (если не указан, используется
                 глобальный синглтон)

    Возвращает:
        Словарь с ключами:
            - question (str): Исходный вопрос
            - generation (str): Финальный ответ
            - sources (list[str]): Список файлов-источников
            - is_grounded (bool): Прошёл ли ответ проверку на галлюцинации
            - retrieve_retry_count (int): Число повторных поисков
            - generate_retry_count (int): Число повторных генераций
            - relevant_chunks_count (int): Число найденных релевантных чанков

    Пример использования:
        result = run_graph("Какие отношения были у Базарова с Одинцовой?")
        print(result["generation"])
        # → Ответ с цитатами из глав, где описаны эти отношения
    """
    global _graph, _indexer

    if indexer is not None:
        _indexer = indexer

    if _graph is None:
        _graph = build_graph()

    # Инициализируем начальное состояние
    initial_state: RAGState = {
        "question": query,
        "rewritten_query": "",
        "documents": [],
        "graded_documents": [],
        "context": "",
        "generation": "",
        "sources": [],
        "retrieve_retry_count": 0,
        "generate_retry_count": 0,
        "is_grounded": False,
    }

    logger.info("Запуск RAG пайплайна для вопроса: %s", query[:100])

    # Выполняем граф
    final_state = _graph.invoke(initial_state)

    logger.info(
        "Пайплайн завершён. Повторов поиска: %d, повторов генерации: %d",
        final_state.get("retrieve_retry_count", 0),
        final_state.get("generate_retry_count", 0)
    )

    return {
        "question": final_state["question"],
        "generation": final_state["generation"],
        "sources": final_state["sources"],
        "is_grounded": final_state.get("is_grounded", False),
        "retrieve_retry_count": final_state.get("retrieve_retry_count", 0),
        "generate_retry_count": final_state.get("generate_retry_count", 0),
        "relevant_chunks_count": len(final_state.get("graded_documents", [])),
    }