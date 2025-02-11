import re
import argparse
import os
import shutil
import spacy
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from text_utils import normalize_text, filter_text  
from spacy.lang.es import Spanish
from typing import Callable

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# def split_documents(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=250,
#         length_function=len,
#         is_separator_regex=True,
#         separators=["\n\n"]  # Separadores personalizados
#     )
#     chunks = text_splitter.split_documents(documents)
    
#     # Aplicar normalización y filtrado a cada chunk
#     for chunk in chunks:
#         chunk.page_content = normalize_text(chunk.page_content)  # Normalizar
#         # chunk.page_content = filter_text(chunk.page_content)     # Filtrar

#     # Mostrar contenido filtrado      
#     print("\n=== Contenido filtrado de los chunks ===")
#     for i, chunk in enumerate(chunks):
#         print(f"\nChunk {i + 1} (Filtrado):")
#         print(chunk.page_content)
#         print("-" * 50)
#     return chunks


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page") + 1
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

# Función para contar tokens usando spaCy
def spacy_token_count(text: str, nlp) -> int:
    """Cuenta el número de tokens en un texto usando spaCy."""
    return len(list(nlp(text)))

def split_documents(documents: list[Document]):
    # Cargar el modelo de spaCy para español
    nlp = spacy.load("es_core_news_sm")

    # Configurar el RecursiveCharacterTextSplitter para usar tokens
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Tamaño del chunk en tokens
        chunk_overlap=0,  # Overlap en tokens
        length_function=lambda text: spacy_token_count(text, nlp),  # Función para contar tokens
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],  # Separadores
    )

    chunks = text_splitter.split_documents(documents)

    # Aplicar normalización y filtrado a cada chunk
    for chunk in chunks:
        chunk.page_content = normalize_text(chunk.page_content)  # Normalizar
        # chunk.page_content = filter_text(chunk.page_content)     # Filtrar

    # Mostrar contenido filtrado      
    print("\n=== Contenido filtrado de los chunks ===")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1} (Filtrado):")
        print(chunk.page_content)
        print("-" * 50)
    return chunks    

if __name__ == "__main__":
    main()