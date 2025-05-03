# agent/tools.py
import os
from typing import List, Union, Optional

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from vector import get_retriever_for_business

# Initialize LLM (configure via environment variables)
llm = ChatOpenAI(model='gpt-4.1')
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

def search_reviews(business_name: str, query: str = "", k: int = 5, rating: Optional[str] = None) -> List[Document]:
    """
    Retrieve up to k reviews for a business, optionally filtered by rating category.

    :param business_name: filter on this metadata field
    :param query: free-text to match within the filtered reviews
    :param k: number of documents to return
    :param rating: 'positive' or 'negative' to filter by star rating
    """
    fetch_k = k if not rating else k * 3
    retriever = get_retriever_for_business(business_name=business_name, score_threshold=0.0, k=k)
    docs = retriever.get_relevant_documents(query or business_name)

    if rating in ("positive", "negative"):
        def keep(d: Document) -> bool:
            try:
                score = float(d.metadata.get("rating", 0))
                return (rating == "positive" and score >= 4) or (rating == "negative" and score <= 2)
            except:
                return False
        filtered = [d for d in docs if keep(d)]
        docs = filtered[:k]
    else:
        docs = docs[:k]

    return docs

def summarize_reviews(business_name: str, k: int = 5, rating: Optional[str] = None) -> str:
    """
    Generate a concise summary over the top-k reviews of a business.

    :param business_name: filter on this metadata field
    :param k: number of documents to include in summary
    :return: summary text
    """
    docs = search_reviews(business_name, query="", k=k, rating=rating)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)


def respond_question(business_name: str, question: str, k: int = 5, score_threshold: float = 0.0) -> str:
    """"
    Answer a specific question about a business, using up to k reviews as context,
    pre-filtered by business without similarity threshold.

    :param business_name: filter on this metadata field
    :param question: user question to answer
    :param k: number of context documents
    :param score_threshold: minimum similarity score threshold (use 0.0 to disable)
    :return: answer text
    """
    # Pre-filter by business, accept all similarity when threshold=0.0
    retriever = get_retriever_for_business(
        business_name=business_name,
        score_threshold=score_threshold,
        k=k
    )
    # Use RetrievalQA chain for Q&A
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return qa.run(question)
