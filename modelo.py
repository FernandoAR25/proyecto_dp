import os
import json
import hashlib
import requests
import re

from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
import faiss

# === Configuración de caché en disco ===
CACHE_DIR = "gutenberg_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(key: str) -> str:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

def load_cache(key: str):
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_cache(key: str, data):
    with open(_cache_path(key), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# === Inicialización del modelo de traducción ===
TRANSLATOR_NAME = "Helsinki-NLP/opus-mt-en-es"
tok_trans = MarianTokenizer.from_pretrained(TRANSLATOR_NAME)
model_trans = MarianMTModel.from_pretrained(TRANSLATOR_NAME)

def translate_to_spanish(text: str) -> str:
    """
    Traduce un texto en inglés a español usando MarianMT.
    """
    batch = tok_trans([text], return_tensors="pt", truncation=True)
    gen = model_trans.generate(**batch)
    return tok_trans.decode(gen[0], skip_special_tokens=True)

# === Funciones de ingestión de Gutenberg ===
GUTENDEX_URL = "https://gutendex.com/books"

def gutenberg_search(query: str, limit: int = 5) -> list[dict]:
    cache_key = f"gutenberg_search:{query}:{limit}"
    cached = load_cache(cache_key)
    if cached is not None:
        return cached
    resp = requests.get(GUTENDEX_URL, params={"search": query})
    resp.raise_for_status()
    data = resp.json()
    books = data.get("results", [])[:limit]
    save_cache(cache_key, books)
    return books


def remove_boilerplate(text: str) -> str:
    start = re.search(r"\\\* START OF (?:THIS|THE) PROJECT GUTENBERG", text)
    if start:
        text = text[start.end():]
    end = re.search(r"\\\* END OF (?:THIS|THE) PROJECT GUTENBERG", text)
    if end:
        text = text[:end.start()]
    return text.strip()


def gutenberg_download_text(book: dict) -> str:
    book_id = book.get("id")
    cache_key = f"gutenberg_text_en:{book_id}"
    cached = load_cache(cache_key)
    if cached is not None:
        return cached
    formats = book.get("formats", {})
    txt_url = formats.get("text/plain; charset=utf-8") or formats.get("text/plain")
    if not txt_url:
        raise ValueError(f"No se encontró versión de texto plano para libro {book_id}")
    resp = requests.get(txt_url)
    resp.raise_for_status()
    text = remove_boilerplate(resp.text)
    save_cache(cache_key, text)
    return text

# === Chunking con solape opcional ===
def sent_tokenize_simple(text: str) -> list[str]:
    return re.split(r'(?<=[.!?])\s+', text)

def chunk_text_simple(text: str, max_words: int = 200, overlap: int = 50) -> list[str]:
    sentences = sent_tokenize_simple(text)
    chunks = []
    current_words = []
    for sent in sentences:
        words = sent.split()
        if len(current_words) + len(words) > max_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:] if overlap < len(current_words) else current_words.copy()
        current_words.extend(words)
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks

# === Pipeline de ingestión + traducción + embeddings + FAISS ===
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def ingest_and_index(queries: list[str], limit: int = 3, max_words: int = 200):
    embeddings: list[np.ndarray] = []
    metadata: list[dict] = []
    chunk_counter = 0

    for query in queries:
        books = gutenberg_search(query, limit)
        for book in books:
            # Descargar texto en inglés
            text_en = gutenberg_download_text(book)
            # Traducir al español
            text_es = translate_to_spanish(text_en)
            # Fragmentar texto traducido
            chunks = chunk_text_simple(text_es, max_words=max_words)

            for chunk in chunks:
                # Generar embedding del chunk en español
                vec = EMBED_MODEL.encode(chunk)
                embeddings.append(vec.astype('float32'))
                metadata.append({
                    'chunk_id': chunk_counter,
                    'book_id': book['id'],
                    'title': book['title'],
                    'text': chunk
                })
                chunk_counter += 1

    emb_array = np.vstack(embeddings)
    d = emb_array.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(emb_array)

    faiss.write_index(index, 'gutenberg_es.index')
    with open('chunk_metadata_es.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Indice FAISS en español creado con {len(metadata)} chunks y guardado como 'gutenberg_es.index'.")

# === Ejecución principal ===
if __name__  == '_main_':
    topics = [
        'World War I', 'Trench warfare', 'Battle of the Somme',
        'Battle of Verdun', 'Battle of Passchendaele', 'Gallipoli campaign',
        'Western Front', 'Eastern Front', 'Naval warfare', 'Air warfare',
        'Central Powers', 'Allied Powers', 'Zimmermann Telegram', 'Battle of Jutland',
        'World War II', 'Battle of Stalingrad', 'D-Day Normandy', 'Operation Overlord',
        'Battle of Britain', 'Operation Barbarossa', 'Battle of Kursk',
        'Battle of Midway', 'Pearl Harbor', 'Blitzkrieg', 'Battle of the Bulge',
        'Holocaust'
    ]
    print("Iniciando ingestión, traducción y creación de índice FAISS en español…")
    ingest_and_index(topics, limit=3)