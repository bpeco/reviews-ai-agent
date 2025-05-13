# ui/app_streamlit.py
import os
import sys
from pathlib import Path

# Ensure project root is in Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from agent.tools import search_reviews, summarize_reviews, respond_question

DEFAULT_K = 5


os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

def main():
    st.set_page_config(page_title="Business Reviews Agent", layout="wide")
    st.title("Business Reviews Agent")

    business = st.text_input("Business Name")
    user_query = st.text_input("Your Query")
    k = st.slider("Documents (k)", 1, 100, DEFAULT_K)

    if st.button("Ask"):
        if not business:
            st.error("Please enter a business name.")
            return
        if not user_query:
            st.error("Please enter a query.")
            return

        # Define tools for the agent, capturing current business and k
        tools = [
            Tool(
                name="search_reviews",
                func=lambda q: "\n\n".join([d.page_content for d in search_reviews(business, q, k)]),
                description="Search reviews for a business using a text query"
            ),
            Tool(
                name="summarize_reviews",
                func=lambda rating: summarize_reviews(
                    business,
                    k,
                    rating.strip().lower() if rating.strip().lower() in ("positive","negative") else None
                ),
                description=(
                    "Summarize reviews for this business. "
                    "Input may be 'positive', 'negative', or empty for all reviews."
                )
            ),
            Tool(
                name="respond_question",
                func=lambda q: respond_question(business, q, k),
                description="Answer a specific question about a business using reviews as context"
            )
        ]

        prefix = (
            "You are a detailed review analyst. When you deliver the final answer, "
            "you MUST produce a multi‐paragraph, in‐depth summary that synthesizes both positives "
            "and negatives. Include concrete examples or short quotes from the reviews, "
            "and explain the significance of each point. Do NOT return a single‐sentence answer."
        )

        llm = ChatOpenAI(model='gpt-4.1')
        agent = initialize_agent(
            tools,
            llm,
            agent="chat-zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={"prefix": prefix}
        )

        with st.spinner("Agent is thinking..."):
            response = agent.run(user_query)

        st.subheader("Agent Response")
        st.write(response)


if __name__ == "__main__":
    main()
