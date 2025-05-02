# main.py
import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from generate_dataset import merge_reviews_with_metadata
from vector import init_db, get_retriever_for_business

# Para LangChain + Llama
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
client = OpenAI()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Prompt extendido que usan ambos modos
PROMPT_TEMPLATE = """
You are an expert in answering questions about restaurants.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}

If you base your answer on the reviews, please provide a summary example of them. Do not make up any information.
Answer the question in a conversational tone, as if you were talking to a friend. Use emojis to make it more engaging.
If you don't have enough information to answer the question, say "I don't know" or "I can't answer that" and provide a reason why.
No preamble.
"""

def ask_openai(question: str, reviews: list[str], model_name: str) -> str:
    """
    Envía el prompt completo a la API de OpenAI a través del cliente oficial.
    """
    reviews_content = "\n".join(reviews)
    content = PROMPT_TEMPLATE.format(reviews=reviews_content, question=question)
    resp = client.responses.create(
        model=model_name,
        input=content
    )
    return resp.output_text

def initialize_llama(model_name: str = 'llama3-8b-8192', temperature: float = 0.9):
    """
    Inicializa ChatGroq (Llama) vía LangChain.
    """
    return ChatGroq(model_name=model_name, temperature=temperature)

def build_chain_llama(model):
    """
    Construye el chain de LangChain usando el prompt extendido.
    """
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt | model

def generate_dataset():
    """
    Fusiona los JSON de reseñas y metadata en un CSV único.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    reviews_path = os.path.join(BASE_DIR, 'data', 'review_data.json')
    metadata_path = os.path.join(BASE_DIR, 'data', 'business_data.json')
    output_csv_path = os.path.join(BASE_DIR, 'data', 'complete_reviews.csv')
    merge_reviews_with_metadata(reviews_path, metadata_path, output_csv_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run review QA via LangChain+Llama or direct OpenAI API'
    )
    parser.add_argument(
        '--mode', choices=['langchain', 'openai'], default='openai',
        help="Choose 'langchain' for Llama or 'openai' for direct OpenAI"
    )
    parser.add_argument(
        '--model', default=None,
        help="Model name: 'gpt-3.5-turbo', 'gpt-4' for OpenAI; 'llama3-8b-8192' for Llama"
    )
    parser.add_argument(
        '--force-init', action='store_true',
        help="Force reindexing of the vector database on startup"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    mode = args.mode
    model_name = args.model or ('gpt-3.5-turbo' if mode == 'openai' else 'llama3-8b-8192')

    if not os.path.exists('data/food_reviews.csv'):
        generate_dataset()

    init_db(force=args.force_init)

    if mode == 'langchain':
        llama = initialize_llama(model_name)
        chain = build_chain_llama(llama)

    while True:
        business_name = input("❓ Which business do you have a question about? (or 'q' to quit): ")
        if business_name.lower() == 'q':
            break
        print(f"\n-   [INFO] >> Searching reviews for {business_name}... 🔎")

        retriever = get_retriever_for_business(business_name, 0.0, 1)

        while True:
            question = input("\n❓ Ask your question (or 'b' to choose another business, 'q' to quit): ")
            if question.lower() == 'q':
                return
            if question.lower() == 'b':
                break

            docs = retriever.invoke(question)
            reviews = [doc.page_content for doc in docs]
            if len(reviews) == 0:
                print(f"\n-   [INFO] >> No relevant reviews found for {business_name} 😔")
            else:
                print(f"\n-   [INFO] >> {len(reviews)} relevant reviews found for {business_name} 💪🏼")
                if mode == 'openai':
                    answer = ask_openai(question, reviews, model_name)
                else:
                    answer = chain.invoke({"reviews": docs, "question": question}).content

                print(answer)

if __name__ == "__main__":
    main()
