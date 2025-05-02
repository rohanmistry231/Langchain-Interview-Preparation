# %% [1. Introduction to Prompts]
# Learn dynamic prompt engineering for retail queries with LangChain.

# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import numpy as np
import nltk

def run_prompts_demo():
    # %% [2. Synthetic Retail Query Data]
    queries = [
        {"product": "TechCorp laptop", "question": "What are its features?"},
        {"product": "TechCorp smartphone", "question": "How long is the battery life?"},
        {"product": "TechCorp tablet", "question": "Is it good for students?"}
    ]
    print("Synthetic Data: Retail customer queries created")
    print(f"Queries: {queries}")

    # %% [3. Dynamic Prompt Engineering]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt = PromptTemplate(
        input_variables=["product", "question"],
        template="You are a retail assistant. For the product {product}, answer: {question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    responses = [chain.run(product=query["product"], question=query["question"]) for query in queries]
    print("Dynamic Prompts: Responses generated")
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: {query['product']} - {query['question']}")
        print(f"Response: {response.strip()}")

    # %% [4. Visualization]
    response_lengths = [len(nltk.word_tokenize(resp)) for resp in responses]
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(queries) + 1), response_lengths, color='blue')
    plt.title("Prompt Response Lengths")
    plt.xlabel("Query")
    plt.ylabel("Word Count")
    plt.savefig("prompts_output.png")
    print("Visualization: Response lengths saved as prompts_output.png")

    # %% [5. Interview Scenario: Prompts]
    """
    Interview Scenario: Prompts
    Q: What’s the role of prompt engineering in LangChain?
    A: Prompt engineering designs structured inputs to guide LLM responses, using templates for dynamic, context-specific queries.
    Key: Improves response relevance and consistency.
    Example: PromptTemplate(input_variables=["product", "question"], template="...")
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_prompts_demo()