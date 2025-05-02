# %% [1. Introduction to Chains]
# Learn sequential workflows with LangChain chains for retail applications.

# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import numpy as np
import nltk

def run_chains_demo():
    # %% [2. Synthetic Retail Query Data]
    queries = [
        "What are the features of the TechCorp laptop?",
        "Compare TechCorp laptops and smartphones.",
        "Is the TechCorp laptop good for gaming?"
    ]
    print("Synthetic Data: Retail customer queries created")
    print(f"Queries: {queries}")

    # %% [3. Single LLMChain]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt = PromptTemplate(
        input_variables=["query"],
        template="You are a retail assistant. Answer the customer query: {query}"
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_key="response")
    
    single_chain_responses = [chain.run(query) for query in queries]
    print("Single LLMChain: Responses generated")
    for i, (query, response) in enumerate(zip(queries, single_chain_responses)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {response.strip()}")

    # %% [4. SequentialChain]
    summary_prompt = PromptTemplate(
        input_variables=["response"],
        template="Summarize the following response in one sentence: {response}"
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")
    
    sequential_chain = SequentialChain(
        chains=[chain, summary_chain],
        input_variables=["query"],
        output_variables=["response", "summary"]
    )
    
    sequential_responses = [sequential_chain({"query": query}) for query in queries]
    print("SequentialChain: Responses and summaries generated")
    for i, (query, result) in enumerate(zip(queries, sequential_responses)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {result['response'].strip()}")
        print(f"Summary: {result['summary'].strip()}")

    # %% [5. Visualization]
    response_lengths = [len(nltk.word_tokenize(resp["response"])) for resp in sequential_responses]
    summary_lengths = [len(nltk.word_tokenize(resp["summary"])) for resp in sequential_responses]
    
    plt.figure(figsize=(8, 4))
    x = np.arange(len(queries))
    plt.bar(x - 0.2, response_lengths, 0.4, label='Response Length', color='blue')
    plt.bar(x + 0.2, summary_lengths, 0.4, label='Summary Length', color='green')
    plt.xticks(x, [f"Query {i+1}" for i in range(len(queries))])
    plt.title("Response and Summary Lengths")
    plt.xlabel("Query")
    plt.ylabel("Word Count")
    plt.legend()
    plt.savefig("chains_output.png")
    print("Visualization: Response and summary lengths saved as chains_output.png")

    # %% [6. Interview Scenario: Chains]
    """
    Interview Scenario: Chains
    Q: How do LangChain chains work for LLM workflows?
    A: Chains combine LLMs with prompts to create sequential workflows, like LLMChain for single tasks or SequentialChain for multi-step processes.
    Key: Modular design enables complex task automation.
    Example: SequentialChain(chains=[LLMChain(...), LLMChain(...)], ...)
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_chains_demo()