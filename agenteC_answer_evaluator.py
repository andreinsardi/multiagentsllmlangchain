# agenteC_answer_evaluator.py

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_loader import shared_llm

# Prompt de avaliação melhorado para aula
eval_prompt = PromptTemplate.from_template("""
Você é um avaliador didático de respostas geradas por um sistema multiagente.

Sua tarefa é avaliar se a resposta abaixo atende adequadamente à pergunta, considerando o contexto fornecido.

Critérios:
- A resposta responde claramente à pergunta?
- A resposta utiliza informações do contexto?
- A resposta evita informações inventadas?
- A resposta tem clareza e estilo didático?

Pergunta: {question}
Resposta: {answer}

Veredito final (em uma frase clara e objetiva, indicando se a resposta foi adequada ou não, e por quê):
""")

output_parser = StrOutputParser()

# Pipeline funcional com shared_llm
eval_chain = eval_prompt | shared_llm | output_parser

# Função Agente C
def agenteC_avaliar(pergunta: str, resposta: str) -> str:
    return eval_chain.invoke({"question": pergunta, "answer": resposta})
