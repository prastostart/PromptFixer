# main.py

from optimizer import PromptOptimizer
from logger import init_logger

def main():
    init_logger()

    model_name = "mosaicml/mpt-1b-redpajama-200b-dolly"  # updated model

    optimizer = PromptOptimizer(model_name=model_name)

    initial_prompt = "Classify this text as positive or negative."
    ground_truth = "positive"

    best_prompt, best_score = optimizer.optimize(
        initial_prompt=initial_prompt,
        ground_truth=ground_truth,
        num_rounds=5
    )

    print("\n===== FINAL RESULTS =====")
    print(f"Best Prompt: {best_prompt}")
    print(f"Best Score: {best_score}")

if __name__ == "__main__":
    main()

