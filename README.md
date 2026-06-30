# 📚 RAG-система по роману «Отцы и дети»

Семантический поиск и вопросно-ответная система по тексту романа И.С. Тургенева «Отцы и дети» с использованием Corrective RAG.

## 🏗️ Архитектура
Пользователь → Continue (IntelliJ) → MCP Server → Corrective RAG → Ollama (LLM)
↓
ChromaDB (векторная БД)

text

### Компоненты:
- **Ollama** — локальный LLM-сервер (модель `qwen2.5:3b`)
- **ChromaDB** — векторное хранилище чанков текста
- **Sentence Transformers** — мультиязычная модель эмбеддингов
- **LangGraph** — граф Corrective RAG с проверкой на галлюцинации
- **FastMCP** — MCP-сервер для интеграции с Continue

### Пайплайн Corrective RAG:
1. **Переписывание вопроса** — оптимизация для векторного поиска
2. **Поиск чанков** — семантический поиск в ChromaDB
3. **Оценка релевантности** — фильтрация нерелевантных чанков
4. **Генерация ответа** — ответ строго на основе контекста
5. **Проверка на галлюцинации** — автоматическая перегенерация при проблемах

---

## 📋 Требования

- Python 3.11+
- [Ollama](https://ollama.com/) (запущена локально на `localhost:11434`)
- IntelliJ IDEA с плагином [Continue](https://continue.dev/)

---

## 🚀 Быстрый старт

### 1. Клонирование и установка

```powershell
# Клонируйте репозиторий
git clone <repo-url>
cd otus-ai-graduation-work

# Установи зависимости
pip install -r requirements.txt
```
2. Установка и запуск Ollama
```powershell
# Установи Ollama с https://ollama.com/download

# Загрузи модель для русского языка
ollama pull qwen2.5:3b

# Запусти Ollama (в отдельном терминале)
ollama serve
```
3. Скачивание текста романа
```powershell
python download_book.py
```
Текст будет сохранён в папку otsi_i_deti/ в виде отдельных .txt файлов по главам.

4. Индексация книги
```powershell
python .\tests\run_indexer.py .\otsi_i_deti\
```
После индексации в папке book_chroma_db_ru/ появится векторная база данных.

5. Проверка поиска
```powershell
python .\tests\test_search.py
```
Должен найти релевантные чанки по тестовым запросам.

6. Проверка Ollama
```powershell
python .\tests\test_ollama.py
```
Должен показать:

```text
✅ Подключение к Ollama работает!
   Модель: qwen2.5:3b
```
7. Запуск MCP-сервера
```powershell
$env:TRANSPORT="streamable-http"; $env:MCP_HOST="localhost"; $env:MCP_PORT="8000"; python server.py
```
Сервер запустится на http://localhost:8000/mcp.

8. Настройка Continue в IntelliJ IDEA
- Открой IntelliJ IDEA

- Открой плагин Continue (иконка в боковой панели)

- Нажми ⚙️ (Settings) → редактировать config.yaml или config.json

- Добавь конфигурацию MCP-сервера:

```yaml
name: Отцы и дети RAG
version: 0.0.1
schema: v1
mcpServers:
  - name: Отцы и дети RAG
    transport: streamable-http
    url: http://localhost:8000/mcp
```
- Перезагрузи Continue (Reload MCP Servers)

# 💬 Примеры запросов
В чате Continue используйте инструменты:

```text
Проиндексируй папку otsi_i_deti
```
```text
Какой статус индекса? (index_status)
```
```text
Кто такой Базаров? Какие у него убеждения?
```
```text
Почему Базаров и Павел Петрович стрелялись на дуэли?
```
```text
Опиши отношения Базарова с Одинцовой
```
```text
Сделай краткое изложение главы chapter_10.txt
```
```text
Найди все упоминания о нигилизме (find_relevant_docs)
```
## 🛠️ Инструменты MCP-сервера

| Инструмент | Описание |
|------------|----------|
| `index_folder` | Индексирует `.txt` файлы глав в векторную БД |
| `ask_question` | Задаёт вопрос с полным RAG-пайплайном |
| `find_relevant_docs` | Ищет чанки без генерации ответа |
| `summarize_document` | Создаёт краткое изложение главы |
| `index_status` | Статистика векторного индекса |
| `llm_status` | Проверка доступности LLM |

# 📁 Структура проекта
```text
otus-ai-graduation-work/
├── config.py              # Конфигурация
├── indexer.py             # Индексация и поиск
├── graph.py               # Corrective RAG пайплайн
├── prompts.py             # Шаблоны запросов
├── server.py              # MCP-сервер
├── download_book.py       # Скрипт скачивания романа
├── requirements.txt       # Зависимости
├── tests/
│   ├── run_indexer.py     # Скрипт индексации
│   ├── test_search.py     # Тест поиска
│   └── test_ollama.py     # Тест Ollama
├── otsi_i_deti/           # Главы романа (.txt)
└── book_chroma_db_ru/     # Векторная БД ChromaDB
```

# 🔧 Конфигурация
Основные настройки в config.py:

```python
# LLM
LLM_MODEL = "qwen2.5:3b"
LLM_BASE_URL = "http://localhost:11434/v1"

# Эмбеддинги
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Чанки
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# Поиск
TOP_K = 5

# Corrective RAG
MAX_RETRIEVE_RETRIES = 2
MAX_GENERATE_RETRIES = 1
```
Все параметры можно переопределить через переменные окружения.