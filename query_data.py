import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from text_utils import normalize_text
from get_embedding_function import get_embedding_function
from langchain_core.output_parsers import StrOutputParser
import random

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Eres un asistente especializado en leyes y propuestas legislativas. Tu tarea es responder preguntas relacionadas con el contenido de propuestas de ley y artículos legales, utilizando únicamente la información proporcionada en el contexto. Sigue estas pautas:

1. **Contexto legal**: Responde como si fueras un experto en leyes, utilizando un lenguaje formal y técnico adecuado para el ámbito jurídico.
2. **Precisión**: Basa tu respuesta estrictamente en el contexto proporcionado. Si no hay información suficiente, indica que no puedes responder con certeza.
3. **Claridad**: Explica los conceptos de manera clara y estructurada, utilizando términos jurídicos correctos.
4. **Formato**: Si es necesario, organiza la respuesta en puntos o párrafos para facilitar la lectura.

Contexto proporcionado:
{context}

---

Pregunta: {question}

Respuesta (en español):
"""

MULTI_QUERY_TEMPLATE = """
    Tu tarea es generar cinco versiones alternativas de la pregunta del usuario para mejorar la recuperación de documentos en una base de datos vectorial.
    Las preguntas se hacen sobre el contexto de documentos legales, asi que las preguntas deben enfocarse a dicho dominio
    Devuelve solo las cinco preguntas, sin texto adicional
    Pregunta original: {question}"""

# def generate_alternative_queries(query_text):
#     """Genera versiones alternativas de la consulta usando el modelo LLM."""
#     prompt_template = ChatPromptTemplate.from_template(MULTI_QUERY_TEMPLATE)
#     prompt = prompt_template.format(question=query_text)
#     model = Ollama(model="llama3.2")
#     output_parser = StrOutputParser()
    
#     alternative_queries = output_parser.parse(model.invoke(prompt))
#     return alternative_queries.split("\n")
def generate_alternative_queries(query_text):

    prompt_perspectives = ChatPromptTemplate.from_template(MULTI_QUERY_TEMPLATE)
    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt_perspectives.format(question=query_text))
    
    # Filtrar solo las cinco preguntas sin texto adicional
    alternate_queries = [line.strip() for line in response_text.split("\n") if line.strip()]
    return alternate_queries[:5]  # Asegura que solo sean 5 preguntas


def get_unique_union(documents):
    """ Obtiene documentos únicos fusionando resultados de varias búsquedas. """
    seen = set()
    unique_docs = []
    for doc in documents:
        doc_content = doc.page_content
        if doc_content not in seen:
            seen.add(doc_content)
            unique_docs.append(doc)
    return unique_docs

def query_rag(query_text: str):
    # Generar preguntas alternas
    alternate_queries = generate_alternative_queries(query_text)
    print("Preguntas alternas generadas:")
    for i, q in enumerate(alternate_queries, 1):
        print(f"{i}. {q}")

    # Normalizar y filtrar las queries antes de la búsqueda
    queries = [normalize_text(q) for q in alternate_queries]
    
    # Preparar la base de datos.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Buscar en la base de datos y fusionar los resultados
    results = []
    for q in queries:
        results.extend(db.similarity_search_with_score(q, k=5))
    
    # Eliminar duplicados manteniendo los 5 más relevantes
    unique_results = list({doc.page_content: (doc, score) for doc, score in results}.values())
    unique_results = sorted(unique_results, key=lambda x: x[1], reverse=True)[:5]
    
    # Preparar el contexto.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in unique_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Configurar el modelo y obtener la respuesta.
    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)
    
    # Formatear y mostrar la respuesta.
    sources = [doc.metadata.get("id", None) for doc, _ in unique_results]
    formatted_response = f"Respuesta: {response_text}\nFuentes: {sources}"
    print(formatted_response)
    
    # Mostrar los chunks y metadatos de las fuentes
    print("\n=== Información detallada de las fuentes ===")
    for i, (doc, score) in enumerate(unique_results):
        print(f"\nChunk {i + 1}:")
        print(f"  - Fuente: {doc.metadata.get('source', 'Desconocida')}")
        print(f"  - Página: {doc.metadata.get('page', 'Desconocida')}")
        print(f"  - Puntaje de similitud: {score:.4f}")
        print(f"  - Contenido del chunk:")
        print(doc.page_content)
        print("-" * 50)
    
    return response_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="El texto de la consulta.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

if __name__ == "__main__":
    main()
