# agenteD_refiner.py

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_loader import shared_llm

# Prompt de refinamento melhorado para aula
refine_prompt = PromptTemplate.from_template("""
Você é um editor literário e professor de literatura.

Sua tarefa é reescrever a resposta abaixo para torná-la mais clara, didática e adequada para um público de estudantes de literatura.

Instruções:
- Melhore a estrutura textual da resposta.
- Deixe a linguagem mais acessível e explicativa.
- Não invente informações além do contexto.
- Mantenha a fidelidade ao texto original.

Pergunta: {question}
Resposta original: {answer}

Resposta refinada (mais clara e didática):
""")

output_parser = StrOutputParser()

# Pipeline funcional com shared_llm
refine_chain = refine_prompt | shared_llm | output_parser

# Função Agente D
def agenteD_refinar(pergunta: str, resposta: str) -> str:
    return refine_chain.invoke({"question": pergunta, "answer": resposta})
