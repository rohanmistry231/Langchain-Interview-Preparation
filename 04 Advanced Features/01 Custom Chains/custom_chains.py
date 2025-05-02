# %% [1. Introduction to Custom Chains]
# Learn to design tailored workflows for retail tasks with LangChain.

# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import numpy as np
import nltk

def run_custom_chains_demo():
    # %% [2. Synthetic Retail Query Data]
    queries = [
        "Describe the TechCorp laptop and suggest a use case.",
        "Explain the TechCorp smartphone features and recommend an accessory.",
        "Detail the TechCorp tablet and propose a target audience."
    ]
    print("Synthetic Data: Retail queries created")
    print(f"Queries: {queries}")

    # %% [3. Custom Chain]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    
    # Chain 1: Describe product
    description_prompt = PromptTemplate(
        input_variables=["query"],
        template="You are a retail assistant. Describe the product in the query: {query}"
    )
    description_chain = LLMChain(llm=llm, prompt=description_prompt, output_key="description")
    
    # Chain 2: Suggest recommendation
    suggestion_prompt = PromptTemplate(
        input_variables=["description"],
        template="Based on the product description: {description}, suggest a use case or recommendation."
    )
    suggestion_chain = LLMChain(llm=llm, prompt=suggestion_prompt, output_key="suggestion")
    
    custom_chain = SequentialChain(
        chains=[description_chain, suggestion_chain],
        input_variables=["query"],
        output_variables=["description", "suggestion"]
    )
    
    responses = [custom_chain({"query": query}) for query in queries]
    print("Custom Chain: Responses generated")
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: {query}")
        print(f"Description: {response['description'].strip()}")
        print(f"Suggestion: {response['suggestion'].strip()}")

    # %% [4. Visualization]
    description_lengths = [len(nltk.word_tokenize(resp["description"])) for resp in responses]
    suggestion_lengths = [len(nltk.word_tokenize(resp["suggestion"])) for resp in responses]
    
    plt.figure(figsize=(8, 4))
    x = np.arange(len(queries))
    plt.bar(x - 0.2, description_lengths, 0.4, label='Description Length', color='blue')
    plt.bar(x + 0.2, suggestion_lengths, 0.4, label='Suggestion Length', color='green')
    plt.xticks(x, [f"Query {i+1}" for i in range(len(queries))])
    plt.title("Custom Chain Output Lengths")
    plt.xlabel("Query")
    plt.ylabel("Word Count")
    plt.legend()
    plt.savefig("custom_chains_output.png")
    print("Visualization: Output lengths saved as custom_chains_output.png")

    # %% [5. Interview Scenario: Custom Chains]
    """
    Interview Scenario: Custom Chains
    Q: How do custom chains handle complex tasks?
    A: Custom chains combine multiple LLMChain instances in a SequentialChain for multi-step workflows tailored to specific tasks.
    Key: Modular design enhances flexibility.
    Example: SequentialChain(chains=[LLMChain(...), LLMChain(...)], ...)
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_custom_chains_demo()