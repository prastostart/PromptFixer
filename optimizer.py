# optimizer.py

import random
from logger import log_prompt
from models import TextModel

class PromptOptimizer:
    def __init__(self, model_name="microsoft/phi-2"):
        self.model_wrapper = TextModel(model_name)
        self.device = self.model_wrapper.device

    def score_prompt(self, prompt, ground_truth, model_output):
        """Dummy scoring function (replace later with real metrics)."""
        return random.uniform(0, 1)

    def reflect_on_prompt(self, prompt, score, feedback=""):
        """Rephrase the prompt based on reflection."""
        # Simple reflection: append feedback to the prompt
        return prompt + " " + feedback if feedback else prompt + " (refined)"

    def generate_output(self, prompt, input_text=""):
        """Generate model output for a given prompt."""
        full_prompt = prompt + " " + input_text if input_text else prompt
        return self.model_wrapper.generate(full_prompt)

    def optimize(self, initial_prompt, ground_truth, num_rounds=5):
        """Optimize the prompt over multiple rounds."""
        current_prompt = initial_prompt
        best_prompt = current_prompt
        best_score = 0

        for round_num in range(1, num_rounds + 1):
            # Candidate transformation
            transformation = "initial" if round_num == 1 else "reflection"

            # Generate output and score
            output = self.generate_output(current_prompt)
            score = self.score_prompt(current_prompt, ground_truth, output)

            # Log the prompt evaluation
            candidate_id = round_num  # simple ID for each candidate
            parent_prompt = current_prompt
            log_prompt(round_num, candidate_id, parent_prompt, transformation, current_prompt, score)

            # Reflect to improve prompt
            current_prompt = self.reflect_on_prompt(current_prompt, score)

            # Update best prompt if score improves
            if score > best_score:
                best_score = score
                best_prompt = current_prompt

            print(f"=== Round {round_num} ===")
            print(f"Prompt: {current_prompt}")
            print(f"Score: {score:.3f}")

        return best_prompt, best_score

