# %% [1. Introduction to Agent Types]
# Learn reactive, planning, and ReAct agents for retail tasks with LangChain.

# Setup: pip install langchain langchain-openai numpy matplotlib
import matplotlib.pyplot as plt
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from collections import Counter
import numpy as np

def run_agent_types_demo():
    # %% [2. Synthetic Retail Query Data]
    query = "Handle a customer request for TechCorp laptop stock and discount."
    print("Synthetic Data: Retail query created")
    print(f"Query: {query}")

    # %% [3. Agent Types Comparison]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    
    def mock_stock_check(product):
        return f"Stock for {product}: 10 units."
    
    def mock_discount_calculator(product):
        return f"Discount for {product}: 15% off."
    
    tools = [
        Tool(name="StockCheck", func=mock_stock_check, description="Check product stock"),
        Tool(name="DiscountCalculator", func=mock_discount_calculator, description="Calculate product discount")
    ]
    
    # Reactive Agent (Zero-Shot ReAct)
    reactive_agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    reactive_response = reactive_agent.run(query)
    
    # Planning Agent (Simulated with more steps)
    planning_agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    planning_response = planning_agent.run(f"Plan and execute: {query}")
    
    print("Agent Types: Responses generated")
    print(f"Reactive Agent Response: {reactive_response}")
    print(f"Planning Agent Response: {planning_response}")

    # %% [4. Visualization]
    response_lengths = [
        len(reactive_response.split()),
        len(planning_response.split())
    ]
    plt.figure(figsize=(8, 4))
    plt.bar(['Reactive', 'Planning'], response_lengths, color=['blue', 'green'])
    plt.title("Agent Response Lengths by Type")
    plt.xlabel("Agent Type")
    plt.ylabel("Word Count")
    plt.savefig("agent_types_output.png")
    print("Visualization: Response lengths saved as agent_types_output.png")

    # %% [5. Interview Scenario: Agent Types]
    """
    Interview Scenario: Agent Types
    Q: What’s the difference between reactive and planning agents?
    A: Reactive agents respond directly to queries, while planning agents break tasks into steps for complex scenarios.
    Key: ReAct combines reasoning and action.
    Example: initialize_agent(tools, llm, agent="zero-shot-react-description")
    """

# Execute the demo
if __name__ == "__main__":
    run_agent_types_demo()