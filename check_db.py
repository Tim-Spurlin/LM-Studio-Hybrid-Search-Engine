import h5py
import sys

db_path = "/run/media/saphyre-solutions/My Passport/universal_knowledge_base.h5"
try:
    with h5py.File(db_path, 'r') as f:
        print("Keys:", list(f.keys()))
        if 'chunks' in f:
            print("Chunks keys:", list(f['chunks'].keys()))
            if 'text' in f['chunks']:
                texts = f['chunks']['text'][:]
                print(f"Total rows: {len(texts)}")
                for i, t in enumerate(texts[:5]):
                    print(f"Row {i}: {t}")
except Exception as e:
    print(f"Error: {e}")
