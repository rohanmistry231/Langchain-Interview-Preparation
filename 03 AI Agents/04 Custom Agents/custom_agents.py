# %% [1. Introduction to Custom Agents]
# Learn to build retail-specific agents with LangChain.

# Setup: pip install langchain langchain-openai numpy matplotlib
import matplotlib.pyplot as plt
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from collections import Counter
import numpy as np

def run_custom_agents_demo():
    # %% [2. Synthetic Retail Query Data]
    queries = [
        "Check inventory for TechCorp laptop.",
        "Suggest a product for a student.",
        "Process a return for TechCorp smartphone."
    ]
    print("Synthetic Data: Retail queries created")
    print(f"Queries: {queries}")

    # %% [3. Custom Agent for Retail]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    
    def mock_inventory_check(query):
        return "Inventory: 10 laptops available."
    
    def mock_product_suggestion(query):
        return "Suggestion: TechCorp Tablet, ideal for students."
    
    def mock_return_process(query):
        return "Return: Processed for TechCorp smartphone."
    
    tools = [
        Tool(name="InventoryCheck", func=mock_inventory_check, description="Check product inventory"),
        Tool(name="ProductSuggestion", func=mock_product_suggestion, description="Suggest a product"),
        Tool(name="ReturnProcess", func=mock_return_process, description="Process a product return")
    ]
    
    custom_agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    responses = [custom_agent.run(query) for query in queries]
    print("Custom Agent: Responses generated")
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {response}")

    # %% [4. Visualization]
    tool_calls = [response.split(":")[0].split()[-1] for response in responses]
    tool_counts = Counter(tool_calls)
    
    plt.figure(figsize=(8, 4))
    plt.bar(tool_counts.keys(), tool_counts.values(), color='orange')
    plt.title("Custom Agent Tool Usage")
    plt.xlabel("Tool")
    plt.ylabel("Count")
    plt.savefig("custom_agents_output.png")
    print("Visualization: Tool usage saved as custom_agents_output.png")

    # %% [5. Interview Scenario: Custom Agents]
    """
    Interview Scenario: Custom Agents
    Q: How do you build a custom agent for retail?
    A: Define task-specific tools and initialize an agent with a reasoning framework like ReAct for retail scenarios.
    Key: Tailor tools to domain needs.
    Example: initialize_agent(tools=[Tool(name="InventoryCheck", ...)], llm, ...)
    """

# Execute the demo
if __name__ == "__main__":
    run_custom_agents_demo()