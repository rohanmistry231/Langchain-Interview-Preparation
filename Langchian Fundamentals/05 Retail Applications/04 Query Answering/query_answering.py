# %% [1. Introduction to Query Answering]
# Learn to answer retail customer queries using RAG with LangChain.

# Setup: pip install langchain langchain-openai faiss-cpu numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import numpy as np
import nltk

def run_query_answering_demo():
    # %% [2. Synthetic Retail Document and Query Data]
    documents = [
        Document(page_content="TechCorp Laptop: 16GB RAM, Intel i7, 512GB SSD, great for gaming.", metadata={"product": "Laptop"}),
        Document(page_content="TechCorp Smartphone: Long battery, vibrant display, ideal for media.", metadata={"product": "Smartphone"}),
        Document(page_content="TechCorp Tablet: Lightweight, 10-hour battery, perfect for students.", metadata={"product": "Tablet"})
    ]
    queries = [
        "What’s the best product for gaming?",
        "Which product has a long battery life?",
        "Is there a lightweight product?"
    ]
    print("Synthetic Data: Retail documents and queries created")
    print(f"Documents: {[doc.metadata['product'] for doc in documents]}")
    print(f"Queries: {queries}")

    # %% [3. RAG-Based Query Answering]
    embeddings = OpenAIEmbeddings(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    vector_store = FAISS.from_documents(documents, embeddings)
    llm = OpenAI(api_key="your-openai-api-key")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2})
    )
    
    rag_responses = [rag_chain.run(query) for query in queries]
    non_rag_responses = [llm(query) for query in queries]
    
    print("Query Answering: Responses generated")
    for i, (query, rag_resp, non_rag_resp) in enumerate(zip(queries, rag_responses, non_rag_responses)):
        print(f"Query {i+1}: {query}")
        print(f"RAG Response: {rag_resp.strip()}")
        print(f"Non-RAG Response: {non_rag_resp.strip()}")

    # %% [4. Visualization]
    rag_lengths = [len(nltk.word_tokenize(resp)) for resp in rag_responses]
    non_rag_lengths = [len(nltk.word_tokenize(resp)) for resp in non_rag_responses]
    
    plt.figure(figsize=(8, 4))
    x = np.arange(len(queries))
    plt.bar(x - 0.2, rag_lengths, 0.4, label='RAG Response', color='blue')
    plt.bar(x + 0.2, non_rag_lengths, 0.4, label='Non-RAG Response', color='red')
    plt.xticks(x, [f"Query {i+1}" for i in range(len(queries))])
    plt.title("RAG vs Non-RAG Response Lengths")
    plt.xlabel("Query")
    plt.ylabel("Word Count")
    plt.legend()
    plt.savefig("query_answering_output.png")
    print("Visualization: Response lengths saved as query_answering_output.png")

    # %% [5. Interview Scenario: Query Answering]
    """
    Interview Scenario: Query Answering
    Q: How does RAG improve query answering?
    A: RAG retrieves relevant documents to provide context, improving accuracy and specificity of LLM responses.
    Key: Combines retrieval and generation.
    Example: RetrievalQA.from_chain_type(llm=llm, retriever=...)
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_query_answering_demo()