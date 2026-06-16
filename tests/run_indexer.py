#!/usr/bin/env python3
"""Запуск индексации романа 'Отцы и дети'"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from indexer import Indexer

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Индексация глав романа')
    parser.add_argument('folder', nargs='?', default='./otsi_i_deti',
                       help='Папка с главами (по умолчанию: ./otsi_i_deti)')
    parser.add_argument('--pattern', default='*.txt',
                       help='Шаблон файлов (по умолчанию: *.txt)')

    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"❌ Папка '{folder}' не найдена!")
        return 1

    # Ищем файлы
    files = list(folder.rglob(args.pattern))

    if not files:
        print(f"❌ В папке '{folder}' нет .txt файлов!")
        return 1

    print(f"📚 Найдено {len(files)} файлов")
    for f in files[:10]:
        print(f"   📄 {f.name}")

    print(f"\n🔄 Запуск индексации...")
    indexer = Indexer()
    result = indexer.index_folder(str(folder), args.pattern)

    print(f"\n✅ Индексация завершена!")
    print(f"   📄 Обработано файлов: {result['files_indexed']}")
    print(f"   📦 Создано чанков: {result['chunks_added']}")

    if result['errors']:
        print(f"   ⚠️ Ошибок: {len(result['errors'])}")
        for error in result['errors'][:5]:
            print(f"      - {error}")

    return 0

if __name__ == '__main__':
    sys.exit(main())