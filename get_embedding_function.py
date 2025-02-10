from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List


# def get_embedding_function():
#     # embeddings = BedrockEmbeddings(
#     #     credentials_profile_name="default", region_name="us-east-1"
#     # )
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     # embeddings = OpenAIEmbeddings(
#     #     model="text-embedding-3-small",  # Modelo gratuito de OpenAI
#     # )
#     return embeddings

class SentenceTransformerEmbeddings(Embeddings):
    # def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
    def __init__(self, model_name: str = 'multi-qa-MiniLM-L6-cos-v1'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Generar embeddings para una lista de textos
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        # Generar embedding para una consulta individual
        return self.model.encode(text).tolist()

def get_embedding_function():
    # Retornar una instancia del wrapper
    return SentenceTransformerEmbeddings()