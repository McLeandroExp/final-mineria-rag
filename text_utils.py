import re
import nltk
from nltk.corpus import stopwords
import unicodedata

# nltk.download('stopwords')

# Palabras vacías en español
spanish_stopwords = set(stopwords.words('spanish'))

# Términos legales que deben preservarse
legal_terms = {
    "artículo", "ley", "inciso", "número", "código", "reglamento", 
    "decreto", "resolución", "contrato", "cláusula", "título", 
    "capítulo", "sección", "apartado", "parágrafo", "literal"
}

def filter_text(text: str) -> str:
    # Tokenizar preservando mayúsculas para términos legales
    words = re.findall(r'\b\w+[´’\w-]*\b', text, flags=re.IGNORECASE)
    
    # Filtrar palabras
    filtered_words = [
        word for word in words
        if (
            word.lower() not in spanish_stopwords  # Eliminar palabras vacías
            or word.lower() in legal_terms  # Preservar términos legales
        ) and len(word) > 2  # Eliminar palabras muy cortas (excepto términos legales)
    ]
    
    return ' '.join(filtered_words)

def normalize_text(text: str) -> str:
    """
    Normaliza el texto para mejorar la coincidencia en búsquedas.
    - Convierte a minúsculas.
    - Elimina puntuación y caracteres especiales.
    - Normaliza acentos y caracteres especiales.
    - Elimina espacios adicionales.
    """
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar puntuación y caracteres especiales
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalizar acentos y caracteres especiales
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Eliminar espacios adicionales
    text = ' '.join(text.split())
    
    return text