import sys
import os
import bs4
from bs4 import BeautifulSoup

try:
    from libzim.reader import Archive
    from libzim.search import Query, Searcher
except ImportError:
    print("libzim not installed. Ensure you run this within the venv.")
    sys.exit(1)

def extract_zim(zim_path, output_dir, max_file_size_mb=5):
    print(f"Opening ZIM archive: {zim_path}")
    zim = Archive(zim_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    current_chunk_idx = 1
    current_text = []
    current_size = 0
    max_size_bytes = max_file_size_mb * 1024 * 1024
    
    total_articles = zim.all_entry_count
    print(f"Total entries in ZIM: {total_articles}")
    
    for i in range(total_articles):
        try:
            entry = zim._get_entry_by_id(i)
            # Only process articles (namespace 'A' or 'C')
            if not entry.get_item().path.startswith(('A/', 'C/')):
                continue
                
            item = entry.get_item()
            content = item.content
            
            # Simple heuristic to check if it's html
            if b'<html' in content[:500].lower() or b'<body' in content[:500].lower():
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                
                if text:
                    title = item.title if item.title else item.path
                    article_text = f"\n\n====================\nARTICLE: {title}\n====================\n{text}"
                    current_text.append(article_text)
                    current_size += len(article_text.encode('utf-8'))
                    
            if current_size >= max_size_bytes:
                out_path = os.path.join(output_dir, f"zim_extract_{current_chunk_idx:05d}.txt")
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write("".join(current_text))
                print(f"Wrote {out_path} ({current_size / 1024 / 1024:.2f} MB)")
                
                current_text = []
                current_size = 0
                current_chunk_idx += 1
                
        except Exception as e:
            pass # Skip broken or unsupported entries
            
    # Write the remaining
    if current_text:
        out_path = os.path.join(output_dir, f"zim_extract_{current_chunk_idx:05d}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("".join(current_text))
        print(f"Wrote {out_path} ({current_size / 1024 / 1024:.2f} MB)")
        
    print(f"Finished extracting {zim_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_zim.py <zim_file_path> <output_dir>")
        sys.exit(1)
        
    extract_zim(sys.argv[1], sys.argv[2])
