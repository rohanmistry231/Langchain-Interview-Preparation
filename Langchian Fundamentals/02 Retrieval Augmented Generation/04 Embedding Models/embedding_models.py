# %% [1. Introduction to Embedding Models]
# Learn to use Hugging Face and OpenAI embeddings for retail RAG with LangChain.

# Setup: pip install langchain langchain-openai langchain-huggingface faiss-cpu numpy matplotlib
import matplotlib.pyplot as plt
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def run_embedding_models_demo():
    # %% [2. Synthetic Retail Document Data and Query]
    documents = [
        Document(page_content="TechCorp Laptop: 16GB RAM, Intel i7, 512GB SSD.", metadata={"product": "Laptop"}),
        Document(page_content="TechCorp Smartphone: Long battery, vibrant display.", metadata={"product": "Smartphone"}),
        Document(page_content="TechCorp Tablet: Lightweight, 10-hour battery.", metadata={"product": "Tablet"})
    ]
    query = "Find a product with a good battery."
    print("Synthetic Data: Retail documents and query created")
    print(f"Documents: {[doc.metadata['product'] for doc in documents]}")
    print(f"Query: {query}")

    # %% [3. Embedding Models Comparison]
    openai_embeddings = OpenAIEmbeddings(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    openai_vector_store = FAISS.from_documents(documents, openai_embeddings)
    hf_vector_store = FAISS.from_documents(documents, hf_embeddings)
    
    openai_retrieved = openai_vector_store.similarity_search(query, k=2)
    hf_retrieved = hf_vector_store.similarity_search(query, k=2)
    
    print("Embedding Models: Documents retrieved")
    print("OpenAI Embeddings:")
    for i, doc in enumerate(openai_retrieved):
        print(f"Retrieved {i+1}: {doc.metadata['product']} - {doc.page_content}")
    print("Hugging Face Embeddings:")
    for i, doc in enumerate(hf_retrieved):
        print(f"Retrieved {i+1}: {doc.metadata['product']} - {doc.page_content}")

    # %% [4. Visualization]
    query_openai_emb = openai_embeddings.embed_query(query)
    query_hf_emb = hf_embeddings.embed_query(query)
    doc_openai_embs = [openai_embeddings.embed_query(doc.page_content) for doc in documents]
    doc_hf_embs = [hf_embeddings.embed_query(doc.page_content) for doc in documents]
    
    openai_similarities = [cosine_similarity([query_openai_emb], [emb])[0][0] for emb in doc_openai_embs]
    hf_similarities = [cosine_similarity([query_hf_emb], [emb])[0][0] for emb in doc_hf_embs]
    
    plt.figure(figsize=(10, 4))
    x = np.arange(len(documents))
    plt.bar(x - 0.2, openai_similarities, 0.4, label='OpenAI Embeddings', color='blue')
    plt.bar(x + 0.2, hf_similarities, 0.4, label='Hugging Face Embeddings', color='green')
    plt.xticks(x, [doc.metadata['product'] for doc in documents])
    plt.title("Embedding Model Similarity Comparison")
    plt.xlabel("Product")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.savefig("embedding_models_output.png")
    print("Visualization: Embedding similarities saved as embedding_models_output.png")

    # %% [5. Interview Scenario: Embedding Models]
    """
    Interview Scenario: Embedding Models
    Q: How do embedding models impact RAG performance?
    A: Embeddings determine retrieval quality; OpenAI offers high accuracy, while Hugging Face models are cost-effective and open-source.
    Key: Model choice balances performance and cost.
    Example: HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    """

# Execute the demo
if __name__ == "__main__":
    run_embedding_models_demo()