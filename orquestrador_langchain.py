# orquestrador_langchain.py

from agenteB_answer_generator import agenteB_responder
from agenteC_answer_evaluator import agenteC_avaliar
from agenteD_refiner import agenteD_refinar
from agenteA_context_retriever import agenteA_buscar
from langchain.memory import ConversationBufferMemory

# Memória da conversação
memory = ConversationBufferMemory()

def executar_fluxo(pergunta, palavra_chave):
    print(f"\n[Usuário]: {pergunta}")
    memory.chat_memory.add_user_message(pergunta)

    # Agente A com palavra-chave
    context = agenteA_buscar(pergunta, top_k=5, palavra_chave=palavra_chave)
    print(f"\n[AgenteA - ContextRetriever]:\n{context[:1000]}...\n")  # Mostra só um pedaço do contexto

    # Agente B
    answer = agenteB_responder(pergunta)
    print(f"\n[AgenteB - AnswerGenerator]:\n{answer}\n")

    # Agente C
    avaliacao = agenteC_avaliar(pergunta, answer)
    print(f"\n[AgenteC - AnswerEvaluator]:\n{avaliacao}\n")

    # Agente D se necessário
    if "não adequada" in avaliacao.lower() or "não responde" in avaliacao.lower():
        resposta_refinada = agenteD_refinar(pergunta, answer)
        print(f"\n[AgenteD - RefinerAgent]:\n{resposta_refinada}\n")
        memory.chat_memory.add_ai_message(resposta_refinada)
    else:
        memory.chat_memory.add_ai_message(answer)

# Loop interativo
if __name__ == "__main__":
    while True:
        question = input("Digite sua pergunta (ou 'sair'): ")
        palavra_chave = input("Digite uma palavra-chave para ajudar na busca (ou deixe em branco): ")
        if question.lower() == "sair":
            break
        
        executar_fluxo(question, palavra_chave)
