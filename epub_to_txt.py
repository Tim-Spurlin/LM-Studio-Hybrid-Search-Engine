import sys
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

def epub_to_text(epub_path, txt_path):
    try:
        book = epub.read_epub(epub_path)
        texts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                if text:
                    texts.append(text)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(texts))
    except Exception as e:
        print(f"Failed to parse EPUB: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python epub_to_txt.py <input.epub> <output.txt>")
        sys.exit(1)
    epub_to_text(sys.argv[1], sys.argv[2])
