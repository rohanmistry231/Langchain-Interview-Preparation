# %% [1. Introduction to Tool Integration]
# Learn to integrate tools for retail tasks with LangChain agents.

# Setup: pip install langchain langchain-openai numpy matplotlib
import matplotlib.pyplot as plt
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from collections import Counter
import numpy as np

def run_tool_integration_demo():
    # %% [2. Synthetic Retail Query Data]
    queries = [
        "Check stock for TechCorp laptop.",
        "Calculate discount for TechCorp smartphone.",
        "Search for TechCorp tablet reviews."
    ]
    print("Synthetic Data: Retail queries created")
    print(f"Queries: {queries}")

    # %% [3. Tool Integration]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    
    def mock_stock_check(product):
        return f"Stock for {product}: 10 units available."
    
    def mock_discount_calculator(product):
        return f"Discount for {product}: 15% off."
    
    def mock_search_reviews(product):
        return f"Reviews for {product}: Mostly positive, 4.5/5 rating."
    
    tools = [
        Tool(name="StockCheck", func=mock_stock_check, description="Check product stock"),
        Tool(name="DiscountCalculator", func=mock_discount_calculator, description="Calculate product discount"),
        Tool(name="SearchReviews", func=mock_search_reviews, description="Search product reviews")
    ]
    
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    responses = [agent.run(query) for query in queries]
    print("Tool Integration: Agent responses generated")
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {response}")

    # %% [4. Visualization]
    tool_calls = [response.split(":")[0].split()[-1] for response in responses]  # Extract tool name
    tool_counts = Counter(tool_calls)
    
    plt.figure(figsize=(8, 4))
    plt.bar(tool_counts.keys(), tool_counts.values(), color='blue')
    plt.title("Tool Call Frequencies")
    plt.xlabel("Tool")
    plt.ylabel("Count")
    plt.savefig("tool_integration_output.png")
    print("Visualization: Tool call frequencies saved as tool_integration_output.png")

    # %% [5. Interview Scenario: Tool Integration]
    """
    Interview Scenario: Tool Integration
    Q: How do agents use tools in LangChain?
    A: Agents use tools to perform specific tasks, selected based on query context via reasoning.
    Key: Tools are defined with functions and descriptions.
    Example: Tool(name="StockCheck", func=mock_stock_check, description="...")
    """

# Execute the demo
if __name__ == "__main__":
    run_tool_integration_demo()