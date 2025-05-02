# %% [1. Introduction to Review Analysis]
# Learn sentiment and topic extraction from retail reviews with LangChain.

# Setup: pip install langchain langchain-openai numpy matplotlib pandas nltk scikit-learn
import matplotlib.pyplot as plt
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from collections import Counter
import numpy as np
import nltk

def run_review_analysis_demo():
    # %% [2. Synthetic Retail Review Data]
    reviews = [
        "The TechCorp laptop is fast and reliable, great for gaming!",
        "TechCorp smartphone has poor battery life, but nice display.",
        "TechCorp tablet is lightweight, but the app selection is limited."
    ]
    print("Synthetic Data: Retail reviews created")
    print(f"Reviews: {reviews}")

    # %% [3. Sentiment and Topic Extraction]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    
    sentiment_prompt = PromptTemplate(
        input_variables=["review"],
        template="You are a retail analyst. Determine the sentiment (Positive, Negative, Neutral) of this review: {review}"
    )
    sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt)
    
    topic_prompt = PromptTemplate(
        input_variables=["review"],
        template="Extract the main topic (e.g., performance, battery, design) of this review: {review}"
    )
    topic_chain = LLMChain(llm=llm, prompt=topic_prompt)
    
    sentiments = [sentiment_chain.run(review=review).strip() for review in reviews]
    topics = [topic_chain.run(review=review).strip() for review in reviews]
    
    print("Review Analysis: Sentiment and topics extracted")
    for i, (review, sentiment, topic) in enumerate(zip(reviews, sentiments, topics)):
        print(f"Review {i+1}: {review}")
        print(f"Sentiment: {sentiment}, Topic: {topic}")

    # %% [4. Visualization]
    sentiment_counts = Counter(sentiments)
    plt.figure(figsize=(8, 4))
    plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color='blue')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig("review_analysis_output.png")
    print("Visualization: Sentiment distribution saved as review_analysis_output.png")

    # %% [5. Interview Scenario: Review Analysis]
    """
    Interview Scenario: Review Analysis
    Q: How is sentiment extracted from reviews?
    A: Sentiment is extracted using LLMs to classify reviews as Positive, Negative, or Neutral based on text content.
    Key: Prompt engineering ensures accurate classification.
    Example: PromptTemplate(...template="Determine the sentiment...")
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_review_analysis_demo()