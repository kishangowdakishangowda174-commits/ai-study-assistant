import streamlit as st
from utils.rag import chunk_text, create_vector_store, retrieve
from groq import Groq
from duckduckgo_search import DDGS

# 🔑 Put your Groq API key here
client = Groq(api_key="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")

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
        st.write(f"Error: {e}")

# ⭐ Extra Feature: Generate Questions
if st.button("Generate Questions"):
    context = retrieve(query)

    prompt = f"Generate 3 study questions from this content:\n{context}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    st.write(response.choices[0].message.content)
