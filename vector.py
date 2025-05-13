import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "db/chroma")
EMBEDDING_FN = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=EMBEDDING_FN
)

def get_retriever_for_business(
    business_name: str,
    score_threshold: float = 0.0,
    k: int = 5
):
    """
    Return a Chroma retriever that:
    - filters by business_name metadata
    - uses a similarity threshold to include all vectors when threshold=0.0
    - returns up to k documents
    """
    return vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "filter": {"business_name": business_name},
            "score_threshold": score_threshold,
            "k": k
        }
    )
