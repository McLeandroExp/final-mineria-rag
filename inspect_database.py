# inspect_database.py
import argparse
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

def inspect_database():
    """
    Función para inspeccionar todos los chunks almacenados en la base de datos Chroma.
    """
    # Cargar la base de datos existente
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Obtener todos los documentos almacenados
    items = db.get(include=["documents", "metadatas"])

    # Mostrar información de cada chunk
    print("\n=== Inspección de la base de datos ===")
    print(f"Número total de chunks almacenados: {len(items['ids'])}")
    for i, (doc_id, document, metadata) in enumerate(zip(items["ids"], items["documents"], items["metadatas"])):
        print(f"\nChunk {i + 1}:")
        print(f"  - ID en la base de datos: {doc_id}")
        print(f"  - Fuente: {metadata.get('source', 'Desconocida')}")
        print(f"  - Página: {metadata.get('page', 'Desconocida')}")
        print(f"  - Contenido del chunk:")
        print(document)
        print("-" * 50)

def query_database(query_text: str, k: int = 5):
    """
    Realiza una consulta en la base de datos y muestra los chunks más relevantes.
    Incluye el número en la base de datos (ID) de cada chunk.
    """
    # Cargar la base de datos existente
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Realizar la consulta
    results = db.similarity_search_with_score(query_text, k=k)

    # Mostrar los resultados
    print(f"\n=== Resultados de la consulta: '{query_text}' ===")
    for i, (doc, score) in enumerate(results):
        print(f"\nChunk {i + 1}:")
        print(f"  - ID en la base de datos: {doc.metadata.get('id', 'Desconocido')}")
        print(f"  - Fuente: {doc.metadata.get('source', 'Desconocida')}")
        print(f"  - Página: {doc.metadata.get('page', 'Desconocida')}")
        print(f"  - Puntaje de similitud: {score:.4f}")
        print(f"  - Contenido del chunk:")
        print(doc.page_content)
        print("-" * 50)

def main():
    # Configurar CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true", help="Inspeccionar todos los chunks en la base de datos.")
    parser.add_argument("--query", type=str, help="Realizar una consulta en la base de datos.")
    args = parser.parse_args()

    if args.inspect:
        inspect_database()  # Inspeccionar la base de datos
    elif args.query:
        query_database(args.query)  # Realizar una consulta
    else:
        print("Usa --inspect para inspeccionar la base de datos o --query <texto> para realizar una consulta.")

if __name__ == "__main__":
    main()        