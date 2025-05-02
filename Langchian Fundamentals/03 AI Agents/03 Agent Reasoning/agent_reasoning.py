# %% [1. Introduction to Agent Reasoning]
# Learn autonomous decision-making for customer support with LangChain agents.

# Setup: pip install langchain langchain-openai numpy matplotlib
import matplotlib.pyplot as plt
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from collections import Counter
import numpy as np

def run_agent_reasoning_demo():
    # %% [2. Synthetic Retail Query Data]
    queries = [
        "Customer asks if TechCorp laptop is in stock.",
        "Customer wants a discount on TechCorp smartphone.",
        "Customer needs help with TechCorp tablet warranty."
    ]
    print("Synthetic Data: Retail customer queries created")
    print(f"Queries: {queries}")

    # %% [3. Agent Reasoning]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    
    def mock_stock_check(query):
        return "Stock: 10 units available."
    
    def mock_discount_offer(query):
        return "Offer: 15% discount applied."
    
    def mock_warranty_info(query):
        return "Warranty: 1-year coverage."
    
    tools = [
        Tool(name="StockCheck", func=mock_stock_check, description="Check product stock"),
        Tool(name="DiscountOffer", func=mock_discount_offer, description="Offer a discount"),
        Tool(name="WarrantyInfo", func=mock_warranty_info, description="Provide warranty details")
    ]
    
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    responses = [agent.run(query) for query in queries]
    print("Agent Reasoning: Responses generated")
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {response}")

    # %% [4. Visualization]
    tool_calls = [response.split(":")[0].split()[-1] for response in responses]
    tool_counts = Counter(tool_calls)
    
    plt.figure(figsize=(8, 4))
    plt.bar(tool_counts.keys(), tool_counts.values(), color='purple')
    plt.title("Agent Decision Tool Usage")
    plt.xlabel("Tool")
    plt.ylabel("Count")
    plt.savefig("agent_reasoning_output.png")
    print("Visualization: Tool usage saved as agent_reasoning_output.png")

    # %% [5. Interview Scenario: Agent Reasoning]
    """
    Interview Scenario: Agent Reasoning
    Q: How does agent reasoning work for retail tasks?
    A: Agents reason by selecting tools based on query context, using LLMs to plan actions autonomously.
    Key: ReAct framework enhances decision-making.
    Example: initialize_agent(tools, llm, agent="zero-shot-react-description")
    """

# Execute the demo
if __name__ == "__main__":
    run_agent_reasoning_demo()