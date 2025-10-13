# optimizer.py
from models import TextModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util

class PromptOptimizer:
    def __init__(self, model_name="mosaicml/mpt-1b-redpajama-200b-dolly", retriever=None):
        self.model_wrapper = TextModel(model_name)
        self.device = self.model_wrapper.device
        self.retriever = retriever  # optional retrieval context
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # semantic embeddings

        # reference answer embedding for BCR
        self.reference_text = (
            "Inflation is the rate at which prices for goods and services increase over time, "
            "reducing purchasing power. Its effect on investments depends on interest rates, risk, and asset type."
        )
        self.ref_embedding = self.embedding_model.encode(self.reference_text, convert_to_tensor=True)

    def score_bcr(self, output_text):
        """
        Semantic BCR: cosine similarity between output and reference answer.
        Returns 0-1 score.
        """
        out_embedding = self.embedding_model.encode(output_text, convert_to_tensor=True)
        score = util.cos_sim(out_embedding, self.ref_embedding).item()
        return round(float(score), 3)

    def reflect_on_prompt(self, prompt, root_cause):
        """
        Refines prompt with optional retrieved context and root cause.
        """
        retrieval_text = ""
        if self.retriever:
            retrieved_docs = self.retriever.retrieve(prompt)
            retrieval_text = "\n".join(retrieved_docs)

        reflection_instruction = (
            f"Refine this financial prompt to fix the issue: {root_cause}. "
            f"Use factual, concise, on-topic content only. {retrieval_text}"
        )
        full_prompt = f"{prompt} {reflection_instruction}"

        new_prompt = self.model_wrapper.generate(full_prompt, max_new_tokens=100, temperature=0.2)
        return new_prompt.strip()

    def optimize(self, initial_prompt, num_rounds=1):
        all_data = []
        best_prompt = initial_prompt
        best_score = 0.0

        for round_num in range(1, num_rounds + 1):
            # optionally add retrieval context
            retrieval_text = ""
            if self.retriever:
                retrieved_docs = self.retriever.retrieve(initial_prompt)
                retrieval_text = "\n".join(retrieved_docs)

            full_prompt = f"{initial_prompt} {retrieval_text}"
            initial_output = self.model_wrapper.generate(full_prompt, temperature=0.2)
            initial_score = self.score_bcr(initial_output)
            print(f"Round {round_num} | Initial BCR Score: {initial_score}")

            root_cause = "Lacks clarity or precision"  # simplified heuristic

            refined_prompts = [self.reflect_on_prompt(initial_prompt, root_cause) for _ in range(10)]

            for idx, prompt in enumerate(refined_prompts, start=1):
                output = self.model_wrapper.generate(prompt, temperature=0.2)
                score = self.score_bcr(output)
                all_data.append({
                    "Round": round_num,
                    "Prompt No.": idx,
                    "Prompt": prompt,
                    "Output": output[:200].replace("\n", " ") + "...",
                    "Root Cause Fixed": root_cause,
                    "BCR Score": score
                })
                if score > best_score:
                    best_score = score
                    best_prompt = prompt

        df = pd.DataFrame(all_data)
        return best_prompt, best_score, df

