# indexador.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import hnswlib

# Importa o TextSplitter recomendado
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Lê o texto
with open("data/dom_casmurro.txt", "r", encoding="latin1") as f:
    texto = f.read()

# 2. Configura o TextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Tamanho alvo do chunk (em caracteres)
    chunk_overlap=100,    # Overlap entre os chunks
)

# 3. Gera os blocos
blocos = splitter.split_text(texto)

# 4. Gera embeddings
print(f"Gerando embeddings para {len(blocos)} blocos...")
modelo = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = modelo.encode(blocos, convert_to_numpy=True)

# 5. Cria o índice vetorial
index = hnswlib.Index(space="cosine", dim=384)
index.init_index(max_elements=len(blocos), ef_construction=200, M=16)
index.add_items(embeddings)

# 6. Salva o índice e os blocos
index.save_index("indice.bin")
pd.DataFrame({"bloco": blocos}).to_csv("blocos.csv", index=False)

print("Indexação concluída com sucesso!")
