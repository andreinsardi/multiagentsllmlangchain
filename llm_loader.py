# llm_loader.py

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# Cria o pipeline UMA VEZ só (compartilhado por todos os agentes)
shared_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
shared_llm = HuggingFacePipeline(pipeline=shared_pipeline)
