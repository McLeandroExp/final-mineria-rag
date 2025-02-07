import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from text_utils import normalize_text, filter_text 
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# Nuevo prompt en español y ajustado al contexto legal
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

def main():
    # Crear CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="El texto de la consulta.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Normalizar y filtrar la query antes de la búsqueda
    normalized_query = normalize_text(query_text)  # Normalizar
    # filtered_query = filter_text(normalized_query) # Filtrar
    
    # Preparar la base de datos.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Buscar en la base de datos.
    results = db.similarity_search_with_score(normalized_query, k=5)

    # Preparar el contexto.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Configurar el modelo y obtener la respuesta.
    model = Ollama(model="llama3.2")
    response_text = model.invoke(prompt)

    # Formatear y mostrar la respuesta.
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Respuesta: {response_text}\nFuentes: {sources}"
    print(formatted_response)

    # Mostrar los chunks y metadatos de las fuentes
    print("\n=== Información detallada de las fuentes ===")
    for i, (doc, score) in enumerate(results):
        print(f"\nChunk {i + 1}:")
        print(f"  - Fuente: {doc.metadata.get('source', 'Desconocida')}")
        print(f"  - Página: {doc.metadata.get('page', 'Desconocida')}")
        print(f"  - Puntaje de similitud: {score:.4f}")
        print(f"  - Contenido del chunk:")
        print(doc.page_content)
        print("-" * 50)

    return response_text

if __name__ == "__main__":
    main()