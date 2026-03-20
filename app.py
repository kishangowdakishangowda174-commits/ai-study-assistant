import sys
import os
import streamlit as st

# 🔧 Robust import for local utils
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

utils_dir = os.path.join(current_dir, "utils")
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from rag import chunk_text, create_vector_store, retrieve
from groq import Groq
from duckduckgo_search import DDGS

# 🔑 Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Please set your GROQ_API_KEY in Streamlit Secrets")
client = Groq(api_key=GROQ_API_KEY)

st.title("AI Study Assistant")

# 🎛️ Response mode
mode = st.radio("Response Mode", ["Concise", "Detailed"])

# 📂 Upload file
uploaded_file = st.file_uploader("Upload your notes (txt file)")

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    chunks = chunk_text(text)
    create_vector_store(chunks)
    st.success("Document processed!")

# 🌐 Web search function
def web_search(query):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        return [r["body"] for r in results]

# ❓ Ask question
query = st.text_input("Ask a question")

if query:
    context = retrieve(query)

    # If no context → use web search
    if not context:
        context = web_search(query)

    instruction = "Answer in 3-4 lines." if mode == "Concise" else "Give a detailed explanation with examples."

    prompt = f"""
    You are a helpful AI Study Assistant.

    Context:
    {context}

    Question:
    {query}

    Instruction:
    {instruction}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.choices[0].message.content
        st.write(answer)

    except Exception as e:
        st.error(f"Error: {e}")

# ⭐ Extra Feature: Generate Questions
if st.button("Generate Questions") and query:
    context = retrieve(query)

    prompt = f"Generate 3 study questions from this content:\n{context}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )

        st.write(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error: {e}")
