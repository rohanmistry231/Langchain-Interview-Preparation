# %% [1. Introduction to Integration with APIs]
# Learn to connect LangChain with retail APIs for task automation.

# Setup: pip install langchain langchain-openai numpy matplotlib
import matplotlib.pyplot as plt
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from collections import Counter
import numpy as np

def run_api_integration_demo():
    # %% [2. Synthetic Retail Query Data]
    queries = [
        "Check TechCorp laptop price via API.",
        "Fetch TechCorp smartphone specs via API.",
        "Get TechCorp tablet availability via API."
    ]
    print("Synthetic Data: Retail queries created")
    print(f"Queries: {queries}")

    # %% [3. API Integration]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    
    def mock_price_api(query):
        return "Price: $999 for TechCorp laptop."
    
    def mock_specs_api(query):
        return "Specs: 8GB RAM, 128GB storage for TechCorp smartphone."
    
    def mock_availability_api(query):
        return "Availability: In stock for TechCorp tablet."
    
    tools = [
        Tool(name="PriceAPI", func=mock_price_api, description="Fetch product price via API"),
        Tool(name="SpecsAPI", func=mock_specs_api, description="Fetch product specs via API"),
        Tool(name="AvailabilityAPI", func=mock_availability_api, description="Check product availability via API")
    ]
    
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    responses = [agent.run(query) for query in queries]
    print("API Integration: Responses generated")
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {response}")

    # %% [4. Visualization]
    api_calls = [response.split(":")[0].split()[-1] for response in responses]
    api_counts = Counter(api_calls)
    
    plt.figure(figsize=(8, 4))
    plt.bar(api_counts.keys(), api_counts.values(), color='blue')
    plt.title("API Call Frequencies")
    plt.xlabel("API")
    plt.ylabel("Count")
    plt.savefig("api_integration_output.png")
    print("Visualization: API call frequencies saved as api_integration_output.png")

    # %% [5. Interview Scenario: Integration with APIs]
    """
    Interview Scenario: Integration with APIs
    Q: How does LangChain integrate with external APIs?
    A: LangChain agents use tools to call APIs, enabling dynamic data retrieval for retail tasks.
    Key: Tools map queries to API functions.
    Example: Tool(name="PriceAPI", func=mock_price_api, description="...")
    """

# Execute the demo
if __name__ == "__main__":
    run_api_integration_demo()