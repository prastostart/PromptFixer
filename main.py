# main.py
from optimizer import PromptOptimizer
from logger import init_logger
import pandas as pd
# from your_retriever_module import SmartRetriever  # uncomment if you want retrieval

def main():
    init_logger()

    model_name = "mosaicml/mpt-1b-redpajama-200b-dolly"
    # retriever = SmartRetriever(docs=your_finance_docs)  # optional
    optimizer = PromptOptimizer(model_name=model_name)  # add retriever=retriever if using

    initial_prompt = "Explain how inflation affects investment returns."
    print(f"\n=== Initial Prompt ===\n{initial_prompt}")

    best_prompt, best_score, df = optimizer.optimize(initial_prompt=initial_prompt, num_rounds=1)

    print("\n=== Generated Prompts and Scores ===")
    print(df.to_string(index=False))

    print("\n===== FINAL RESULTS =====")
    print(f"Best Prompt:\n{best_prompt}")
    print(f"Best Score (BCR): {best_score}")

    df.to_csv("prompt_logs.csv", index=False)
    with open("best_prompts.csv", "w", encoding="utf-8") as f:
        f.write(f"best_prompt,best_score\n{best_prompt},{best_score}\n")

if __name__ == "__main__":
    main()

