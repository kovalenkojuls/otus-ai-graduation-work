import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import re

# Настройки
BOOK_URL = "https://ilibrary.ru/text/96/index.html"
OUTPUT_DIR = "otsi_i_deti"

def download_book():
    """Скачивает книгу по главам"""

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print(f"📥 Скачиваем книгу...")

    # 1. Получаем страницу
    response = requests.get(BOOK_URL)
    response.encoding = 'windows-1251'
    content = response.text

    soup = BeautifulSoup(content, 'html.parser')

    # 2. Находим название книги
    title_tag = soup.find('h1') or soup.find('title')
    book_name = title_tag.get_text(strip=True) if title_tag else "Книга"
    print(f"📚 {book_name}")

    # 3. Находим все ссылки на странице
    all_links = soup.find_all('a', href=True)
    print(f"\n🔍 Всего ссылок: {len(all_links)}")

    # Показываем содержание
    print("\n📑 Содержание книги:")
    chapters = []

    for link in all_links:
        href = link.get('href', '')
        text = link.get_text(strip=True)

        if text and len(text) > 3:
            # Ищем ссылки на главы (обычно содержат /text/ID/)
            if '/text/' in href and href.count('/') >= 3:
                full_url = f"https://ilibrary.ru{href}" if href.startswith('/') else href
                if full_url != BOOK_URL:
                    chapters.append((full_url, text))
                    print(f"  {len(chapters)}. {text}")

    print(f"\n📚 Найдено глав: {len(chapters)}")

    if not chapters:
        print("❌ Главы не найдены!")
        return

    # 4. Скачиваем каждую главу
    print(f"\n📥 Скачивание глав...")
    success = 0

    for i, (url, title) in enumerate(chapters, 1):
        try:
            # Получаем главу
            response = requests.get(url)
            response.encoding = 'windows-1251'  # ✅ Правильная кодировка
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            # Ищем контейнер с текстом
            content_div = (
                soup.find('div', class_='text') or
                soup.find('div', id='text') or
                soup.find('div', class_='content')
            )

            if content_div:
                # Удаляем навигацию
                for elem in content_div.find_all(['nav', 'script', 'style', 'div'],
                                                   class_=['nav', 'navigation']):
                    elem.decompose()

                text = content_div.get_text()

                # Очищаем текст
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = '\n'.join(line.strip() for line in text.split('\n'))
                text = text.strip()

                if len(text) > 100:
                    # Сохраняем
                    safe_title = re.sub(r'[<>:"/\\|?*]', '', title)[:50]
                    filename = Path(OUTPUT_DIR) / f"{i:03d}_{safe_title}.txt"

                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(text)

                    # Показываем начало текста
                    preview = text[:100].replace('\n', ' ')
                    print(f"  ✓ {i}/{len(chapters)}: {title[:40]}")
                    print(f"     {preview}...")
                    success += 1
                else:
                    print(f"  ⚠️ {i}: {title[:40]} - текст короткий")
            else:
                print(f"  ⚠️ {i}: {title[:40]} - контейнер не найден")

            time.sleep(0.5)

        except Exception as e:
            print(f"  ❌ {i}: {title[:40]} - {e}")

    print(f"\n{'='*50}")
    print(f"✅ Готово! Сохранено {success}/{len(chapters)} глав")
    print(f"📁 Папка: {OUTPUT_DIR}")

    # Показываем пример
    files = sorted(Path(OUTPUT_DIR).glob("*.txt"))
    if files:
        print(f"\n📖 Проверка - первый файл:")
        with open(files[0], 'r', encoding='utf-8') as f:
            print(f.read()[:300])
        print("...")

    return success


if __name__ == "__main__":
    download_book()