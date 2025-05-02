# %% [1. Introduction to Memory]
# Learn contextual conversation history with LangChain memory for retail interactions.

# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import numpy as np
import nltk

def run_memory_demo():
    # %% [2. Synthetic Retail Conversation Data]
    conversation = [
        "Tell me about TechCorp laptops.",
        "Which one is best for gaming?",
        "What’s the price of that model?"
    ]
    print("Synthetic Data: Retail customer conversation created")
    print(f"Conversation: {conversation}")

    # %% [3. Conversational Memory]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt = PromptTemplate(
        input_variables=["history", "query"],
        template="You are a retail assistant. Given the conversation history:\n{history}\nAnswer the query: {query}"
    )
    memory = ConversationBufferMemory(return_messages=True)
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    
    responses = []
    for query in conversation:
        response = chain.run(query=query)
        responses.append(response)
        memory.save_context({"query": query}, {"output": response})
    
    print("Memory: Conversational responses generated")
    for i, (query, response) in enumerate(zip(conversation, responses)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {response.strip()}")

    # %% [4. Visualization]
    history_lengths = [len(nltk.word_tokenize(memory.buffer_as_str)) for _ in range(len(conversation))]
    response_lengths = [len(nltk.word_tokenize(resp)) for resp in responses]
    
    plt.figure(figsize=(8, 4))
    x = np.arange(len(conversation))
    plt.plot(x, history_lengths, marker='o', label='History Length', color='blue')
    plt.plot(x, response_lengths, marker='x', label='Response Length', color='green')
    plt.xticks(x, [f"Query {i+1}" for i in range(len(conversation))])
    plt.title("Conversation History and Response Lengths")
    plt.xlabel("Query")
    plt.ylabel("Word Count")
    plt.legend()
    plt.savefig("memory_output.png")
    print("Visualization: History and response lengths saved as memory_output.png")

    # %% [5. Interview Scenario: Memory]
    """
    Interview Scenario: Memory
    Q: How does memory enhance conversational agents in LangChain?
    A: Memory stores conversation history, enabling context-aware responses for coherent interactions.
    Key: Types like ConversationBufferMemory retain query-response pairs.
    Example: ConversationBufferMemory(return_messages=True)
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_memory_demo()