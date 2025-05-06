import os
from sentence_transformers import SentenceTransformer
import numpy as np


def embed_txt_file(input_txt: str,
                   output_txt: str,
                   model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Lee un archivo .txt con un chunk por línea, genera embeddings para cada línea usando
    un modelo de Sentence-Transformers, y guarda todos los embeddings en un archivo .txt.

    Cada línea del archivo de salida será un vector separado por espacios.
    """
    # Cargar modelo
    model = SentenceTransformer(model_name)

    # Leer todas las líneas (cada una es un chunk)
    with open(input_txt, 'r', encoding='utf-8') as f:
        textos = [line.strip() for line in f if line.strip()]

    # Generar embeddings
    embeddings = model.encode(
        textos,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Guardar embeddings
    with open(output_txt, 'w', encoding='utf-8') as fout:
        for vec in embeddings:
            fout.write(' '.join(str(x) for x in vec.tolist()) + '\n')

    print(f"✅ {len(embeddings)} embeddings guardados en '{output_txt}'")


if __name__ == '__main__':
    INPUT_TXT = "chunks_preprocesados_WW1.txt"      # <-- tu archivo limpio
    OUTPUT_TXT = "embeddings_consolidados_WW1.txt"  # <-- archivo de salida

    embed_txt_file(INPUT_TXT, OUTPUT_TXT)
