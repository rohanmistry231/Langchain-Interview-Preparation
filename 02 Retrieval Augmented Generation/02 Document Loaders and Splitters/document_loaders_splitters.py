# %% [1. Introduction to Document Loaders and Splitters]
# Learn to process and split retail documents with LangChain.

# Setup: pip install langchain langchain-openai numpy matplotlib pandas nltk
import matplotlib.pyplot as plt
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
import nltk

def run_document_loaders_splitters_demo():
    # %% [2. Synthetic Retail Document Data]
    document = Document(
        page_content="TechCorp Laptop Manual: The laptop features a 16GB RAM, Intel i7 processor, and 512GB SSD. It has a vibrant 15-inch display and a 10-hour battery life. Ideal for professionals and gamers.",
        metadata={"product": "TechCorp Laptop"}
    )
    print("Synthetic Data: Retail document created")
    print(f"Document: {document.metadata['product']} - {document.page_content[:50]}...")

    # %% [3. Document Splitting]
    text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator=".")
    chunks = text_splitter.split_documents([document])
    print("Document Splitter: Document chunks created")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk.page_content}")

    # %% [4. Visualization]
    chunk_lengths = [len(nltk.word_tokenize(chunk.page_content)) for chunk in chunks]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(chunks) + 1), chunk_lengths, color='green')
    plt.title("Document Chunk Lengths")
    plt.xlabel("Chunk")
    plt.ylabel("Word Count")
    plt.savefig("document_loaders_splitters_output.png")
    print("Visualization: Chunk lengths saved as document_loaders_splitters_output.png")

    # %% [5. Interview Scenario: Document Loaders and Splitters]
    """
    Interview Scenario: Document Loaders and Splitters
    Q: How do document splitters optimize RAG?
    A: Splitters break large documents into smaller chunks for efficient retrieval and processing by LLMs.
    Key: Balances chunk size with context retention.
    Example: CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_document_loaders_splitters_demo()