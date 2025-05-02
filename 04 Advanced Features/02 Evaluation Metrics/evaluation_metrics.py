# %% [1. Introduction to Evaluation Metrics]
# Learn BLEU, ROUGE, and custom metrics for retail response quality with LangChain.

# Setup: pip install langchain langchain-openai numpy matplotlib nltk rouge-score
import matplotlib.pyplot as plt
from langchain.llms import OpenAI
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np

def run_evaluation_metrics_demo():
    # %% [2. Synthetic Retail Query and Reference Data]
    queries = [
        "Describe the TechCorp laptop features.",
        "Explain the TechCorp smartphone battery.",
        "Detail the TechCorp tablet use case."
    ]
    references = [
        "The TechCorp laptop has 16GB RAM, Intel i7, and 512GB SSD.",
        "The TechCorp smartphone offers a long-lasting battery with vibrant display.",
        "The TechCorp tablet is lightweight, ideal for students and professionals."
    ]
    print("Synthetic Data: Retail queries and references created")
    print(f"Queries: {queries}")

    # %% [3. Response Generation and Evaluation]
    llm = OpenAI(api_key="your-openai-api-key")  # Replace with your OpenAI API key
    responses = [llm(query) for query in queries]
    
    bleu_scores = []
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    for ref, resp in zip(references, responses):
        # BLEU Score
        ref_tokens = [ref.split()]
        resp_tokens = resp.split()
        bleu = sentence_bleu(ref_tokens, resp_tokens)
        bleu_scores.append(bleu)
        
        # ROUGE Score
        rouge = scorer.score(ref, resp)
        rouge_scores.append(rouge['rouge1'].fmeasure)
    
    print("Evaluation Metrics: Scores calculated")
    for i, (query, resp, bleu, rouge) in enumerate(zip(queries, responses, bleu_scores, rouge_scores)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {resp.strip()}")
        print(f"BLEU: {bleu:.2f}, ROUGE-1: {rouge:.2f}")

    # %% [4. Visualization]
    plt.figure(figsize=(8, 4))
    x = np.arange(len(queries))
    plt.bar(x - 0.2, bleu_scores, 0.4, label='BLEU', color='blue')
    plt.bar(x + 0.2, rouge_scores, 0.4, label='ROUGE-1', color='red')
    plt.xticks(x, [f"Query {i+1}" for i in range(len(queries))])
    plt.title("BLEU and ROUGE Scores")
    plt.xlabel("Query")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("evaluation_metrics_output.png")
    print("Visualization: Scores saved as evaluation_metrics_output.png")

    # %% [5. Interview Scenario: Evaluation Metrics]
    """
    Interview Scenario: Evaluation Metrics
    Q: What are BLEU and ROUGE used for?
    A: BLEU measures n-gram overlap, ROUGE evaluates recall and precision for text similarity, both used to assess response quality.
    Key: Metrics quantify LLM performance.
    Example: sentence_bleu([ref.split()], resp.split())
    """

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_evaluation_metrics_demo()