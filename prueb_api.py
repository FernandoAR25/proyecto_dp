import os
import json
import hashlib
import requests
from nltk.tokenize import sent_tokenize

# 1. Configuración de cache en disco
CACHE_DIR = "wiki_cache"
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
    path = _cache_path(key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 2. Búsqueda de títulos relevantes
def wiki_search(query: str, limit: int = 5) -> list[str]:
    cache_key = f"search:{query}:{limit}"
    cached = load_cache(cache_key)
    if cached is not None:
        return cached

    resp = requests.get(
        "https://es.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list":   "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json"
        }
    ).json()
    titles = [item["title"] for item in resp["query"]["search"]]
    save_cache(cache_key, titles)
    return titles

# 3. Extracción de texto de un artículo
def wiki_extract(title: str) -> str:
    cache_key = f"extract:{title}"
    cached = load_cache(cache_key)
    if cached is not None:
        return cached

    resp = requests.get(
        "https://es.wikipedia.org/w/api.php",
        params={
            "action":    "query",
            "prop":      "extracts",
            "explaintext": True,
            "titles":    title,
            "format":    "json"
        }
    ).json()
    page = next(iter(resp["query"]["pages"].values()))
    text = page.get("extract", "")
    save_cache(cache_key, text)
    return text

# 4. Chunking en fragmentos de ~250 palabras (aprox. 3–5 oraciones)
def chunk_text(text: str, max_words: int = 250) -> list[str]:
    sentences = sent_tokenize(text, language="spanish")
    chunks = []
    current = []
    count = 0

    for sent in sentences:
        words = len(sent.split())
        if count + words > max_words and current:
            chunks.append(" ".join(current))
            current = []
            count = 0
        current.append(sent)
        count += words

    if current:
        chunks.append(" ".join(current))
    return chunks

# 5. Flujo de ejemplo
if __name__ == "__main__":
    query = "Inicio de la segunda guerra mundial"
    print(f"Buscando artículos para: {query!r}")
    titles = wiki_search(query, limit=3)
    print("Títulos encontrados:", titles)

    for title in titles:
        print(f"\n--- Extrayendo y chunking: {title!r} ---")
        text = wiki_extract(title)
        chunks = chunk_text(text)
        print(f"  → {len(chunks)} chunks generados.")
        # Muestra los primero 2 chunks como ejemplo
        for i, c in enumerate(chunks[:2], 1):
            preview = c[:200].replace("\n"," ")
            print(f"    Chunk {i}: {preview}…\n")
