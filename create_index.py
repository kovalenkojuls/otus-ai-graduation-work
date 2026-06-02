from pathlib import Path
from datetime import datetime, timezone
import hashlib
import chardet
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Config:
    CHROMA_DB_PATH = "./book_chroma_db_ru"
    COLLECTION_NAME = "fathers_and_sons_ru"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K = 5

cfg = Config()

def read_file_with_encoding(path: Path) -> str:
    raw = path.read_bytes()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return raw.decode("utf-8", errors="replace")


def chunk_id(source: str, chunk_index: int) -> str:
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()

class BookIndexer:
    """Индексатор с поддержкой русского языка"""

    def __init__(self, db_path: str = None):
        db_path = db_path or cfg.CHROMA_DB_PATH

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Удаляем существующую коллекцию, если есть
        try:
            self._client.delete_collection(cfg.COLLECTION_NAME)
            print(f"🗑️  Старая коллекция '{cfg.COLLECTION_NAME}' удалена")
        except:
            pass

        self._collection = self._client.get_or_create_collection(
            name=cfg.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )

        self._last_indexed_at = None
        self._indexed_files = []

    def index_book_folder(self, folder_path: str) -> dict:
        root = Path(folder_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Папка не найдена: {root}")

        files = sorted(root.glob("*.txt"))
        if not files:
            files = sorted(root.rglob("*.txt"))

        print(f"\n📚 Найдено файлов: {len(files)}")

        chunks_added = 0
        errors = []

        for file_path in files:
            try:
                new_chunks = self._index_file(file_path)
                chunks_added += new_chunks
                self._indexed_files.append(str(file_path))
                print(f"  ✓ {file_path.name}: {new_chunks} чанков")
            except Exception as e:
                print(f"  ❌ {file_path.name}: {e}")
                import traceback
                traceback.print_exc()  # Покажет полную ошибку
                errors.append(f"{file_path}: {e}")

        self._last_indexed_at = datetime.now(timezone.utc)

        return {
            "files_indexed": len(files) - len(errors),
            "chunks_added": chunks_added,
            "errors": errors
        }

    def get_status(self) -> dict:
        return {
            "total_chunks": self._collection.count(),
            "files_count": len(set(self._indexed_files)),
            "collection_name": cfg.COLLECTION_NAME,
            "last_indexed_at": self._last_indexed_at
        }

    def _index_file(self, file_path: Path) -> int:
        content = read_file_with_encoding(file_path)
        if not content.strip():
            return 0

        chapter_name = file_path.stem

        enhanced_content = f"Глава: {chapter_name}\n\n{content}"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.CHUNK_SIZE,
            chunk_overlap=cfg.CHUNK_OVERLAP,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                ""
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
            return 0

        ids = []
        texts = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(chunk_id(str(file_path), i))
            texts.append(chunk.page_content)
            metadatas.append({
                "source": str(file_path),
                "chunk_index": i,
                "chapter": chapter_name,
                "char_count": len(chunk.page_content)
            })

        # Добавляем батчами для производительности
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self._collection.upsert(
                ids=ids[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

        return len(chunks)

def create_index():
    """Создание индекса для романа 'Отцы и дети'"""

    print("=" * 70)
    print("СОЗДАНИЕ ИНДЕКСА 'ОТЦЫ И ДЕТИ' С РУССКОЙ МОДЕЛЬЮ")
    print("=" * 70)

    print(f"\n📥 Индексация...")
    print(f"   Модель: sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    print(f"   Размер чанка: {cfg.CHUNK_SIZE}")
    print(f"   Перекрытие: {cfg.CHUNK_OVERLAP}")
    print(f"   База данных: {cfg.CHROMA_DB_PATH}")

    indexer = BookIndexer()

    try:
        result = indexer.index_book_folder("./otsi_i_deti")

        print(f"\n{'='*70}")
        print(f"✅ ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
        print(f"{'='*70}")
        print(f"   Файлов обработано: {result['files_indexed']}")
        print(f"   Чанков создано: {result['chunks_added']}")

        if result['errors']:
            print(f"   ❌ Ошибок: {len(result['errors'])}")
            for error in result['errors']:
                print(f"      - {error}")

        status = indexer.get_status()
        print(f"\n📊 Статус коллекции:")
        print(f"   Коллекция: {status['collection_name']}")
        print(f"   Всего чанков: {status['total_chunks']}")
        print(f"   Уникальных файлов: {status['files_count']}")

        print(f"\n💾 Индекс сохранен в: {cfg.CHROMA_DB_PATH}")
        print(f"✅ Готово!\n")

    except FileNotFoundError:
        print("❌ Папка './otsi_i_deti' не найдена!")
        print("   Поместите файлы романа в папку 'otsi_i_deti' и запустите снова")
    except Exception as e:
        print(f"❌ Ошибка при индексации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_index()