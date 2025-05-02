# %% [1. Introduction to Vector Stores]
# Learn Faiss-based document retrieval for retail applications with LangChain.

# Setup: pip install langchain langchain-openai faiss-cpu numpy matplotlib
import matplotlib.pyplot as plt
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def run_vector_stores_demo():
    # %% [2. Synthetic Retail Document Data]
    documents = [
        Document(page_content="TechCorp Laptop: 16GB RAM, Intel i7, 512GB SSD.", metadata={"product": "Laptop"}),
        Document(page_content="TechCorp Smartphone: Long battery, vibrant display.", metadata={"product": "Smartphone"}),
        Document(page_content="TechCorp Tablet: Lightweight, 10-hour battery.", metadata={"product": "Tablet"})
    ]
    query = "Find a laptop with good performance."
    print("Synthetic Data: Retail documents and query created")
    print(f"Documents: {[doc.metadata['product'] for doc in documents]}")
    print(f"Query: {query}")

    # %% [3. Faiss Vector Store]
    embeddings = OpenAIEmbeddings(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    vector_store = FAISS.from_documents(documents, embeddings)
    retrieved_docs = vector_store.similarity_search(query, k=2)
    print("Vector Store: Documents retrieved")
    for i, doc in enumerate(retrieved_docs):
        print(f"Retrieved {i+1}: {doc.metadata['product']} - {doc.page_content}")

    # %% [4. Visualization]
    query_embedding = embeddings.embed_query(query)
    doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in documents]
    similarities = [cosine_similarity([query_embedding], [emb])[0][0] for emb in doc_embeddings]
    
    plt.figure(figsize=(8, 4))
    plt.bar([doc.metadata['product'] for doc in documents], similarities, color='blue')
    plt.title("Document Similarity to Query")
    plt.xlabel("Product")
    plt.ylabel("Cosine Similarity")
    plt.savefig("vector_stores_output.png")
    print("Visualization: Document similarities saved as vector_stores_output.png")

    # %% [5. Interview Scenario: Vector Stores]
    """
    Interview Scenario: Vector Stores
    Q: What’s the role of vector stores in RAG?
    A: Vector stores index document embeddings for efficient similarity-based retrieval, enhancing LLM responses.
    Key: Faiss enables fast nearest-neighbor search.
    Example: FAISS.from_documents(documents, embeddings)
    """

# Execute the demo
if __name__ == "__main__":
    run_vector_stores_demo()