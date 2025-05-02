# %% [1. Introduction to Chatbots]
# Learn to build conversational agents for retail customer support with LangChain.

# Setup: pip install langchain langchain-openai numpy matplotlib nltk
import matplotlib.pyplot as plt
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import numpy as np
import nltk

def run_chatbots_demo():
    # %% [2. Synthetic Retail Conversation Data]
    conversation = [
        "What are the features of the TechCorp laptop?",
        "Is it good for gaming?",
        "What’s the price?"
    ]
    print("Synthetic Data: Retail customer conversation created")
    print(f"Conversation: {conversation}")

    # %% [3. Chatbot with Memory]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    prompt = PromptTemplate(
        input_variables=["history", "query"],
        template="You are a retail assistant. Given the conversation history:\n{history}\nAnswer: {query}"
    )
    memory = ConversationBufferMemory(return_messages=True)
    chatbot = LLMChain(llm=llm, prompt=prompt, memory=memory)
    
    responses = []
    for query in conversation:
        response = chatbot.run(query=query)
        responses.append(response)
        memory.save_context({"query": query}, {"output": response})
    
    print("Chatbot: Responses generated")
    for i, (query, response) in enumerate(zip(conversation, responses)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {response.strip()}")

    # %% [4. Visualization]
    response_lengths = [len(nltk.word_tokenize(resp)) for resp in responses]
    history_lengths = [len(nltk.word_tokenize(memory.buffer_as_str)) for _ in range(len(conversation))]
    
    plt.figure(figsize=(8, 4))
    x = np.arange(len(conversation))
    plt.plot(x, response_lengths, marker='o', label='Response Length', color='blue')
    plt.plot(x, history_lengths, marker='x', label='History Length', color='green')
    plt.xticks(x, [f"Query {i+1}" for i in range(len(conversation))])
    plt.title("Chatbot Response and History Lengths")
    plt.xlabel("Query")
    plt.ylabel("Word Count")
    plt.legend()
    plt.savefig("chatbots_output.png")
    print("Visualization: Response and history lengths saved as chatbots_output.png")

    # %% [5. Interview Scenario: Chatbots]
    """
    Interview Scenario: Chatbots
    Q: How do chatbots use memory in LangChain?
    A: Chatbots use memory to retain conversation history, ensuring context-aware responses for coherent interactions.
    Key: ConversationBufferMemory stores query-response pairs.
    Example: ConversationBufferMemory(return_messages=True)
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_chatbots_demo()