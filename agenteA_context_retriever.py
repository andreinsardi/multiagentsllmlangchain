import hnswlib
import pandas as pd
from sentence_transformers import SentenceTransformer

# Carrega os blocos salvos no indexador
df = pd.read_csv("blocos.csv")

# Carrega o índice hnswlib
index = hnswlib.Index(space="cosine", dim=384)
index.load_index("indice.bin")

# Carrega o modelo SentenceTransformer
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Função Agente A: busca os k blocos mais relevantes para a pergunta
def agenteA_buscar(pergunta: str, top_k: int = 3, palavra_chave: str = None) -> str:
    # Gera embedding da pergunta
    emb = modelo.encode([pergunta], convert_to_numpy=True)

    # Ajusta k para não passar do número de blocos no índice
    k = min(top_k, index.get_current_count())
    if k == 0:
        return "Nenhum bloco encontrado."

    # Busca os k blocos mais similares
    idxs, _ = index.knn_query(emb, k=k)

    # Retorna os blocos concatenados, com quebra entre blocos
    blocos = [df.iloc[i].bloco for i in idxs[0]]

    # Novo: filtro para blocos que contenham a palavra-chave digitada
    if palavra_chave and palavra_chave.strip():
        blocos_filtrados = [b for b in blocos if palavra_chave.lower() in b.lower()]

        # Se nenhum bloco com a palavra-chave, volta para fallback sem filtro
        if len(blocos_filtrados) > 0:
            blocos = blocos_filtrados

    return "\n\n".join(blocos)
