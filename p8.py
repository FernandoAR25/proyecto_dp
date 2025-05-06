"""
crawler_wiki_ww2.py
Reúne rápidamente la mayoría de los artículos relevantes a la Segunda Guerra Mundial
usando categorías + descarga asíncrona de extractos.
"""

import os, json, hashlib, asyncio, aiohttp
from collections import deque
from typing import Set, List
from nltk.tokenize import sent_tokenize

# ---------- Caché disco (idéntico) ----------
CACHE_DIR = "wiki_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _path(k):
    import hashlib, os
    return os.path.join(CACHE_DIR,
        f"{hashlib.sha256(k.encode()).hexdigest()}.json")

def load(k):
    p = _path(k)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:          # ← especifica UTF‑8
            return json.load(f)

def save(k, data):
    with open(_path(k), "w", encoding="utf-8") as f:  # ← especifica UTF‑8
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- API helpers ----------
API = "https://es.wikipedia.org/w/api.php"
HEAD = {"User-Agent": "ww2-scraper/1.0 (roberto zamora)"}

async def _api(session, **params):
    params |= {"format": "json"}
    async with session.get(API, params=params, headers=HEAD) as r:
        r.raise_for_status()
        return await r.json()

# ---------- 1. Recolectar títulos vía categorías ----------
async def crawl_categories(root: str, depth: int = 2,
                           session=None) -> Set[str]:
    """
    Devuelve un set de títulos (namespace 0) bajo `root` hasta `depth` niveles.
    """
    key = f"catcrawl:{root}:{depth}"
    if (c := load(key)): return set(c)

    own = False
    if session is None:
        own = True
        session = aiohttp.ClientSession()

    titles, q = set(), deque([(root, 0)])
    while q:
        cat, lvl = q.popleft()
        cmcontinue = ""
        while True:
            data = await _api(session,
                              action="query",
                              list="categorymembers",
                              cmtitle=f"Categoría:{cat}",
                              cmtype="page|subcat",
                              cmlimit=500,
                              cmcontinue=cmcontinue)
            for it in data["query"]["categorymembers"]:
                if it["ns"] == 0:                    # artículo
                    titles.add(it["title"])
                elif it["ns"] == 14 and lvl < depth: # sub‑categoría
                    sub = it["title"].split(":",1)[1]
                    q.append((sub, lvl+1))
            cmcontinue = data.get("continue", {}).get("cmcontinue", "")
            if not cmcontinue:
                break

    if own: await session.close()
    save(key, sorted(titles))
    return titles

# ---------- 2. Descargar extractos en paralelo ----------
async def fetch_extracts(titles: List[str], session, sem, max_try=2):
    results = {}
    async def worker(title):
        cache_k = f"extract:{title}"
        if (c := load(cache_k)):
            results[title] = c; return
        for attempt in range(max_try):
            try:
                async with sem:
                    data = await _api(
                        session,
                        action="query",
                        prop="extracts",
                        explaintext=1,                 # bandera “texto plano”
                        titles=title,
                        exlimit=1,
                        exsectionformat="plain"
                    )                  
                page = next(iter(data["query"]["pages"].values()))
                text = page.get("extract", "")
                save(cache_k, text)
                results[title] = text
                return
            except Exception as e:
                if attempt + 1 == max_try:
                    print("falló:", title, e)

    workers = [asyncio.create_task(worker(t)) for t in titles]
    await asyncio.gather(*workers)
    return results

# ---------- 3. Chunking ----------
def chunk_text(text, max_words=400):
    sents, chunks, cur, cnt = sent_tokenize(text, "spanish"), [], [], 0
    for s in sents:
        w = len(s.split())
        if cnt + w > max_words and cur:
            chunks.append(" ".join(cur)); cur, cnt = [], 0
        cur.append(s); cnt += w
    if cur: chunks.append(" ".join(cur))
    return chunks

# ---------- 4. Uso de alto nivel ----------
async def harvest_ww2(depth=2, concurrency=30):
    root = "Segunda Guerra Mundial"
    async with aiohttp.ClientSession() as sess:
        titles = await crawl_categories(root, depth, sess)
        print(f"{len(titles):,} artículos detectados")
        extracts = await fetch_extracts(
            list(titles), sess, sem=asyncio.Semaphore(concurrency))
    # Opcional: generar los chunks y guardarlos aparte
    total_chunks = 0
    for t, tx in extracts.items():
        cks = chunk_text(tx)
        total_chunks += len(cks)
        save(f"chunks:{t}", cks)
    print(f"{total_chunks:,} chunks generados y cacheados")

# ---------- 5. Ejecución ----------
if __name__ == "__main__":
    asyncio.run(harvest_ww2(depth=2, concurrency=40))
