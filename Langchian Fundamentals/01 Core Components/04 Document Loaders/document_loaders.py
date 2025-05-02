# %% [1. Introduction to Document Loaders]
# Learn to process retail data with LangChain document loaders.

# Setup: pip install langchain langchain-openai numpy matplotlib pandas nltk
import matplotlib.pyplot as plt
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import numpy as np
import nltk

def run_document_loaders_demo():
    # %% [2. Synthetic Retail Document Data]
    documents = [
        Document(
            page_content="TechCorp Laptop Manual: Features include a 16GB RAM, Intel i7 processor, and 512GB SSD.",
            metadata={"product": "TechCorp Laptop"}
        ),
        Document(
            page_content="TechCorp Smartphone Review: Excellent battery life, vibrant display, but average camera.",
            metadata={"product": "TechCorp Smartphone"}
        ),
        Document(
            page_content="TechCorp Tablet Guide: Lightweight design, 10-hour battery, ideal for students.",
            metadata={"product": "TechCorp Tablet"}
        )
    ]
    print("Synthetic Data: Retail documents created")
    print(f"Documents: {[doc.metadata['product'] for doc in documents]}")

    # %% [3. Document Processing]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt = PromptTemplate(
        input_variables=["content"],
        template="You are a retail assistant. Summarize the document: {content}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    summaries = [chain.run(content=doc.page_content) for doc in documents]
    print("Document Loaders: Summaries generated")
    for i, (doc, summary) in enumerate(zip(documents, summaries)):
        print(f"Document {i+1}: {doc.metadata['product']}")
        print(f"Summary: {summary.strip()}")

    # %% [4. Visualization]
    document_lengths = [len(nltk.word_tokenize(doc.page_content)) for doc in documents]
    summary_lengths = [len(nltk.word_tokenize(summary)) for summary in summaries]
    
    plt.figure(figsize=(8, 4))
    x = np.arange(len(documents))
    plt.bar(x - 0.2, document_lengths, 0.4, label='Document Length', color='blue')
    plt.bar(x + 0.2, summary_lengths, 0.4, label='Summary Length', color='green')
    plt.xticks(x, [doc.metadata['product'] for doc in documents])
    plt.title("Document and Summary Lengths")
    plt.xlabel("Product")
    plt.ylabel("Word Count")
    plt.legend()
    plt.savefig("document_loaders_output.png")
    print("Visualization: Document and summary lengths saved as document_loaders_output.png")

    # %% [5. Interview Scenario: Document Loaders]
    """
    Interview Scenario: Document Loaders
    Q: How do document loaders process external data in LangChain?
    A: Document loaders parse text from various sources into Document objects, enabling LLM processing for tasks like summarization.
    Key: Support diverse formats like PDFs, CSVs, or raw text.
    Example: Document(page_content="...", metadata={...})
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_document_loaders_demo()