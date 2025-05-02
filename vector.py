# vector.py
import os
import shutil
from tqdm import tqdm
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Directorio donde Chroma persiste la base
db_location = './chroma_langchain_db'
# Modelo de embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Placeholder para la instancia de Chroma (será creada en init_db)
vector_store = None

def init_db(force: bool = False):
    """
    Inicializa la base de datos de Chroma:
      - Si force=True, borra y recrea la carpeta entera.
      - Si no existe la carpeta, la crea.
      - (Re)instancia vector_store tras asegurar el directorio.
      - Si la colección está vacía o force, indexa todas las reviews.
    """
    global vector_store

    # 1) Forzar eliminación de la carpeta si corresponde
    if force and os.path.exists(db_location):
        shutil.rmtree(db_location)

    # 2) Asegurar existencia de la carpeta
    os.makedirs(db_location, exist_ok=True)

    # 3) (Re)crear la instancia de Chroma apuntando al directorio limpio
    vector_store = Chroma(
        collection_name="business_reviews",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    # 4) Contar cuántos docs hay ya indexados
    existing = vector_store._collection.count()
    if force or existing == 0:
        # Leer todas las reviews
        df = pd.read_csv('data/complete_reviews.csv')
        documents, ids = [], []
        print(f"Iniciando indexación de {len(df)} documentos...")
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Indexing documents"):
            review_text = row.get('review') or ""
            response_text = row.get('response') or ""
            page_content = f"Review: {review_text} | Response: {response_text}"
            doc = Document(
                page_content=page_content,
                metadata={
                    "business_name": row.get("business_name"),
                    "review_rating": row.get('rating'),
                    "avg_business_rating": row.get('avg_rating'),
                    "num_of_reviews": row.get('num_of_reviews')
                },
                id=str(i)
            )
            documents.append(doc)
            ids.append(str(i))
        print("📦 Persistiendo documentos en Chroma DB…")
        vector_store.add_documents(documents, ids=ids)
        print("✅ Chroma DB initialization complete.")
    else:
        print("✅ Chroma DB ya inicializada. Omitiendo indexación.")

def get_retriever_for_business(business_name: str, score_threshold: float = 0.6, k: int = 5):
    """
    Retorna un Retriever de LangChain que:
      - Filtra solo docs con metadata.business_name == business_name
      - Aplica score_threshold de similitud
      - Devuelve hasta k resultados
    """
    if vector_store is None:
        raise ValueError("Vector store no inicializado. Llama a init_db() primero.")
    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "filter": {"business_name": business_name},
            "score_threshold": score_threshold,
            "k": k
        }
    )
