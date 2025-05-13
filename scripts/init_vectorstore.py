# scripts/init_vectorstore.py
#!/usr/bin/env python3
import os
import shutil
import click
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm


@click.command()
@click.option("--data-path", default="data/reviews.csv", help="Path to the CSV file")
@click.option("--persist-dir", default="db/chroma", help="Directory for Chroma persistence")
@click.option("--drop", is_flag=True, help="Remove existing index before initializing")
def init_vectorstore(data_path: str, persist_dir: str, drop: bool):
    if drop and os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)
        click.echo(f"Index removed: {persist_dir}")

    df = pd.read_csv(data_path)
    docs = []
    for _, row in tqdm(df.iterrows()):
        review = row['review'] if pd.notna(row['review']) else ""
        response = row['response'] if pd.notna(row['response']) else ""
        content = f"Review: {review} | Response: {response}"
        metadata = {
            "business_name": row["business_name"],
            "rating": row["rating"],
            "avg_rating": row["avg_rating"],
            "num_of_reviews": row["num_of_reviews"]
        }
        docs.append(Document(page_content=content, metadata=metadata))

    click.echo(f"Start RecursiveCharacterTextSplitter...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    click.echo(f"Start Spliting...")
    docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    click.echo(f"Start Persisting...")
    vectordb.persist()
    click.echo(f"Vectorstore initialized at: {persist_dir}")

if __name__ == "__main__":
    init_vectorstore()