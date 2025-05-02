# %% [1. Introduction to RAG Pipeline]
# Learn to enhance LLM responses with RAG for retail queries using LangChain.

# Setup: pip install langchain langchain-openai faiss-cpu numpy matplotlib
import matplotlib.pyplot as plt
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import numpy as np
import nltk

def run_rag_pipeline_demo():
    # %% [2. Synthetic Retail Document Data and Query]
    documents = [
        Document(page_content="TechCorp Laptop: 16GB RAM, Intel i7, 512GB SSD, ideal for gaming.", metadata={"product": "Laptop"}),
        Document(page_content="TechCorp Smartphone: Long battery, vibrant display.", metadata={"product": "Smartphone"}),
        Document(page_content="TechCorp Tablet: Lightweight, 10-hour battery.", metadata={"product": "Tablet"})
    ]
    query = "What’s the best product for gaming?"
    print("Synthetic Data: Retail documents and query created")
    print(f"Documents: {[doc.metadata['product'] for doc in documents]}")
    print(f"Query: {query}")

    # %% [3. RAG Pipeline]
    embeddings = OpenAIEmbeddings(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    vector_store = FAISS.from_documents(documents, embeddings)
    llm = OpenAI(api_key="your-openai-api-key")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2})
    )
    
    rag_response = rag_chain.run(query)
    non_rag_response = llm(query)  # Direct LLM response without RAG
    print("RAG Pipeline: Responses generated")
    print(f"RAG Response: {rag_response.strip()}")
    print(f"Non-RAG Response: {non_rag_response.strip()}")

    # %% [4. Visualization]
    response_lengths = [
        len(nltk.word_tokenize(rag_response)),
        len(nltk.word_tokenize(non_rag_response))
    ]
    plt.figure(figsize=(8, 4))
    plt.bar(['RAG', 'Non-RAG'], response_lengths, color=['blue', 'red'])
    plt.title("RAG vs Non-RAG Response Lengths")
    plt.xlabel("Response Type")
    plt.ylabel("Word Count")
    plt.savefig("rag_pipeline_output.png")
    print("Visualization: Response lengths saved as rag_pipeline_output.png")

    # %% [5. Interview Scenario: RAG Pipeline]
    """
    Interview Scenario: RAG Pipeline
    Q: How does RAG improve LLM responses?
    A: RAG retrieves relevant documents to provide context, improving accuracy and relevance over standalone LLMs.
    Key: Combines retrieval with generation.
    Example: RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_rag_pipeline_demo()