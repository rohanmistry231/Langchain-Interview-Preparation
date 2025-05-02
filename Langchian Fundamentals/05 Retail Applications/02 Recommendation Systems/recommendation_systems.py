# %% [1. Introduction to Recommendation Systems]
# Learn to build product recommendation systems using embeddings with LangChain.

# Setup: pip install langchain langchain-openai faiss-cpu numpy matplotlib sklearn
import matplotlib.pyplot as plt
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def run_recommendation_systems_demo():
    # %% [2. Synthetic Retail Product Data]
    products = [
        Document(page_content="TechCorp Laptop: 16GB RAM, Intel i7, 512GB SSD, ideal for gaming.", metadata={"product": "Laptop"}),
        Document(page_content="TechCorp Smartphone: Long battery, vibrant display, great for media.", metadata={"product": "Smartphone"}),
        Document(page_content="TechCorp Tablet: Lightweight, 10-hour battery, perfect for students.", metadata={"product": "Tablet"})
    ]
    user_query = "Recommend a product for a student."
    print("Synthetic Data: Retail products and user query created")
    print(f"Products: {[doc.metadata['product'] for doc in products]}")
    print(f"User Query: {user_query}")

    # %% [3. Recommendation System]
    embeddings = OpenAIEmbeddings(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    vector_store = FAISS.from_documents(products, embeddings)
    recommended_docs = vector_store.similarity_search(user_query, k=2)
    
    print("Recommendation System: Products recommended")
    for i, doc in enumerate(recommended_docs):
        print(f"Recommendation {i+1}: {doc.metadata['product']} - {doc.page_content}")

    # %% [4. Visualization]
    query_embedding = embeddings.embed_query(user_query)
    product_embeddings = [embeddings.embed_query(doc.page_content) for doc in products]
    similarities = [cosine_similarity([query_embedding], [emb])[0][0] for emb in product_embeddings]
    
    plt.figure(figsize=(8, 4))
    plt.bar([doc.metadata['product'] for doc in products], similarities, color='green')
    plt.title("Product Recommendation Similarities")
    plt.xlabel("Product")
    plt.ylabel("Cosine Similarity")
    plt.savefig("recommendation_systems_output.png")
    print("Visualization: Recommendation similarities saved as recommendation_systems_output.png")

    # %% [5. Interview Scenario: Recommendation Systems]
    """
    Interview Scenario: Recommendation Systems
    Q: How do embeddings power recommendation systems?
    A: Embeddings convert product descriptions into vectors, enabling similarity-based matching for personalized recommendations.
    Key: Cosine similarity identifies relevant products.
    Example: FAISS.from_documents(products, embeddings)
    """

# Execute the demo
if __name__ == "__main__":
    run_recommendation_systems_demo()