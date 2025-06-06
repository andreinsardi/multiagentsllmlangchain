# agenteB_answer_generator.py

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_loader import shared_llm
from agenteA_context_retriever import agenteA_buscar

# Prompt melhorado para aula
prompt_template = PromptTemplate.from_template("""
Você é um especialista em literatura brasileira e profundo conhecedor da obra "Dom Casmurro", de Machado de Assis.

Sua tarefa é responder claramente à pergunta do usuário com base no contexto fornecido abaixo. 
Caso o contexto não contenha informações suficientes, você deve dizer explicitamente: 
"Não há uma resposta direta no contexto fornecido."

Instruções:
- Responda de forma objetiva, didática e bem estruturada.
- Utilize exclusivamente as informações do contexto.
- Não invente detalhes que não estejam no texto.

Contexto:
{context}

Pergunta:
{question}

Resposta (em até 3 parágrafos claros e diretos):
""")

output_parser = StrOutputParser()

# Pipeline funcional com shared_llm
qa_chain = prompt_template | shared_llm | output_parser

# Função Agente B
MAX_CONTEXT_CHARS = 1500  # ou 2000 chars, depende do modelo que você está usando

def agenteB_responder(pergunta: str) -> str:
    contexto = agenteA_buscar(pergunta, top_k=5)
    contexto_truncado = contexto[:MAX_CONTEXT_CHARS]  # truncamento seguro

    return qa_chain.invoke({"question": pergunta, "context": contexto_truncado})

