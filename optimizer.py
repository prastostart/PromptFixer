# optimizer.py

from models import TextModel
from logger import log_prompt
import random

class PromptOptimizer:
    def __init__(self, model_name="mosaicml/mpt-7b-instruct"):
        self.model_wrapper = TextModel(model_name)
        self.device = self.model_wrapper.device

    def score_prompt(self, prompt, ground_truth, model_output):
        """
        Use simple accuracy: 1 if model output contains ground_truth, else 0.
        """
        return 1.0 if ground_truth.lower() in model_output.lower() else 0.0

    def reflect_on_prompt(self, prompt, score, feedback=""):
        """
        LLM-based reflection/rephrasing.
        For now, we append a simple instruction for refinement.
        """
        reflection_instruction = "Refine this prompt for clarity and accuracy."
        full_prompt = f"{prompt} {reflection_instruction}"
        new_prompt = self.model_wrapper.generate(full_prompt, max_length=100)
        return new_prompt

    def generate_output(self, prompt, input_text=""):
        full_prompt = prompt + " " + input_text if input_text else prompt
        return self.model_wrapper.generate(full_prompt)

    def optimize(self, initial_prompt, ground_truth, num_rounds=5):
        current_prompt = initial_prompt
        best_prompt = current_prompt
        best_score = -1

        for round_num in range(1, num_rounds + 1):
            # Candidate transformation
            transformation = "initial" if round_num == 1 else "reflection"

            # Generate output and score
            output = self.generate_output(current_prompt)
            score = self.score_prompt(current_prompt, ground_truth, output)

            # Log evaluation
            candidate_id = round_num
            parent_prompt = current_prompt
            log_prompt(round_num, candidate_id, parent_prompt, transformation, current_prompt, score)

            # Rephrase prompt using LLM
            current_prompt = self.reflect_on_prompt(current_prompt, score)

            # Update best prompt
            if score > best_score:
                best_score = score
                best_prompt = current_prompt

            print(f"=== Round {round_num} ===")
            print(f"Prompt: {current_prompt}")
            print(f"Output: {output}")
            print(f"Score: {score}")

        return best_prompt, best_score

