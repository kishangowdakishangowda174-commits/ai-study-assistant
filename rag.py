import faiss
import numpy as np
from models.embeddings import get_embedding

documents = []
index = None

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


def create_vector_store(text_chunks):
    global index, documents

    documents = text_chunks

    embeddings = [get_embedding(chunk) for chunk in text_chunks]
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)


def retrieve(query, k=3):
    global index, documents

    if index is None:
        return []

    query_vector = np.array([get_embedding(query)]).astype("float32")

    distances, indices = index.search(query_vector, k)

    results = []
    for i in indices[0]:
        if i < len(documents):
            results.append(documents[i])

    return results
