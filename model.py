import torch
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration

# === Cargar modelos ===
print("📦 Cargando modelos...")
model_embed = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-base")
model_t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# === Cargar embeddings y chunks ===
with open("embeddings_consolidados_WW2.txt", "r", encoding="utf-8") as f:
    embeddings = [list(map(float, line.strip().split())) for line in f]
embeddings = torch.tensor(embeddings, dtype=torch.float32)

with open("chunks_preprocesados_WW2.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

assert len(chunks) == len(embeddings), "Los embeddings no coinciden con los chunks."

# === Función para generar la respuesta con T5 ===
def responder_con_t5(pregunta, contexto):
    prompt = f"Pregunta: {pregunta} Contexto: {contexto} Respuesta:"
    inputs = tokenizer_t5(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model_t5.generate(**inputs, max_length=100, num_beams=4)
    return tokenizer_t5.decode(outputs[0], skip_special_tokens=True)

# === Loop del chatbot ===
print("\n🤖 Chat estilo RAG (MiniLM + T5). Escribe 'salir' para terminar.\n")

while True:
    pregunta = input("Tú: ")
    if pregunta.lower() in ["salir", "exit", "quit"]:
        break

    emb_pregunta = model_embed.encode(pregunta, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(emb_pregunta, embeddings)[0]
    idx_top = torch.argmax(sims)
    chunk = chunks[idx_top]

    respuesta = responder_con_t5(pregunta, chunk)
    print("\n💬 Respuesta:")
    print(respuesta)
    print("-" * 60)
