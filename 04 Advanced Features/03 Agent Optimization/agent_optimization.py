# %% [1. Introduction to Agent Optimization]
# Learn to optimize agent performance and latency for retail tasks with LangChain.

# Setup: pip install langchain langchain-openai numpy matplotlib
import matplotlib.pyplot as plt
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
import time
import numpy as np

def run_agent_optimization_demo():
    # %% [2. Synthetic Retail Query Data]
    queries = [
        "Check TechCorp laptop stock.",
        "Calculate TechCorp smartphone discount.",
        "Search TechCorp tablet reviews."
    ]
    print("Synthetic Data: Retail queries created")
    print(f"Queries: {queries}")

    # %% [3. Agent Optimization]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    
    def mock_stock_check(query):
        return "Stock: 10 units."
    
    def mock_discount_calculator(query):
        return "Discount: 15% off."
    
    def mock_search_reviews(query):
        return "Reviews: 4.5/5 rating."
    
    tools = [
        Tool(name="StockCheck", func=mock_stock_check, description="Check product stock"),
        Tool(name="DiscountCalculator", func=mock_discount_calculator, description="Calculate discount"),
        Tool(name="SearchReviews", func=mock_search_reviews, description="Search reviews")
    ]
    
    # Non-Optimized Agent
    non_optimized_agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    non_optimized_times = []
    for query in queries:
        start = time.time()
        non_optimized_agent.run(query)
        non_optimized_times.append(time.time() - start)
    
    # Optimized Agent (less verbose, limited tool calls)
    optimized_agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False, max_iterations=3)
    optimized_times = []
    for query in queries:
        start = time.time()
        optimized_agent.run(query)
        optimized_times.append(time.time() - start)
    
    print("Agent Optimization: Execution times recorded")
    for i, (query, non_opt_time, opt_time) in enumerate(zip(queries, non_optimized_times, optimized_times)):
        print(f"Query {i+1}: {query}")
        print(f"Non-Optimized: {non_opt_time:.2f}s, Optimized: {opt_time:.2f}s")

    # %% [4. Visualization]
    plt.figure(figsize=(8, 4))
    x = np.arange(len(queries))
    plt.bar(x - 0.2, non_optimized_times, 0.4, label='Non-Optimized', color='red')
    plt.bar(x + 0.2, optimized_times, 0.4, label='Optimized', color='green')
    plt.xticks(x, [f"Query {i+1}" for i in range(len(queries))])
    plt.title("Agent Execution Times")
    plt.xlabel("Query")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig("agent_optimization_output.png")
    print("Visualization: Execution times saved as agent_optimization_output.png")

    # %% [5. Interview Scenario: Agent Optimization]
    """
    Interview Scenario: Agent Optimization
    Q: How do you optimize agent performance?
    A: Limit iterations, reduce verbosity, and streamline tool calls to lower latency while maintaining accuracy.
    Key: Balances speed and quality.
    Example: initialize_agent(..., max_iterations=3, verbose=False)
    """

# Execute the demo
if __name__ == "__main__":
    run_agent_optimization_demo()