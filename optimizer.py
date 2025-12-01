from models import TextModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
import torch

class PromptOptimizer:
    def __init__(self):
        # We rely on models.py to handle the Qwen GGUF loading automatically
        self.model_wrapper = TextModel()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---------------- 0. Intent Classifier ----------------
    def _extract_intent(self, prompt):
        system_prompt = "You are an expert Intent Classifier."
        query = f"""
        User Prompt: "{prompt}"
        
        Task: Identify the core user intent in 3-6 words.
        Example: "Explain gravity simply" or "Write Python binary search code".
        
        Output ONLY the category string.
        """
        try:
            intent = self.model_wrapper.generate(
                prompt=query, 
                system_instruction=system_prompt, 
                max_new_tokens=30, 
                temperature=0.1
            ).strip()
            return intent
        except:
            return "General factual response"

    # ---------------- 1. The Judge (Un-Anchored & Incremental) ----------------
    def compute_llm_score(self, prompt, output, original_intent):
        if self._is_refusal(output): 
            return 0.0, "Refusal detected."

        redundancy_penalty = 1.0
        if self._check_repetition(output): 
            redundancy_penalty = 0.5

        system_prompt = "You are a precise AI Evaluator."
        
        judge_query = f"""
        ### CONTEXT
        User Intent: "{original_intent}"
        Current Prompt: "{prompt}"
        AI Output: "{output}"
        
        ### SCORING RUBRIC
        - 0.0: Refusal, Hallucination, or Completely Wrong.
        - 0.5: Correct answer, but basic/standard. (PASSING)
        - 0.6: Slightly better structure, tone, or detail. (IMPROVED)
        - 0.7: Good, clear, formatted well. (GOOD)
        - 0.8: Very strong, excellent formatting, covers all nuances. (GREAT)
        - 0.9: Exceptional, creative, perfect. (NEAR PERFECT)
        - 1.0: Flawless. Impossible to improve.
        
        ### TASK
        Evaluate the AI Output.
        If it is "Correct" but could be structured better, give 0.5 or 0.6.
        If it is "Good", give 0.7.
        
        ### RESPONSE FORMAT
        You must format your answer exactly like this:
        Reasoning: [1 sentence analysis]
        Score: [0.0-1.0]
        """
        
        try:
            eval_text = self.model_wrapper.generate(
                prompt=judge_query, 
                system_instruction=system_prompt,
                max_new_tokens=150, 
                temperature=0.1
            ).strip()
            
            match = re.search(r"Score:\s*(0\.\d+|1\.0|0|1)", eval_text, re.IGNORECASE)
            raw_score = float(match.group(1)) if match else 0.5
            
            if "Reasoning:" in eval_text:
                reasoning = eval_text.split("Score:")[0].replace("Reasoning:", "").strip()
            else:
                reasoning = eval_text.replace("\n", " ")[:100]

        except Exception as e:
            raw_score = 0.5
            reasoning = "Error in evaluation."

        final_score = round(raw_score * redundancy_penalty, 4)
        return final_score, reasoning

    # ---------------- 2. The Analyst ----------------
    def diagnose_root_cause(self, prompt, output, score, reasoning):
        if score > 0.95: return "The output is excellent."
        
        system_prompt = "You are an expert AI Consultant."
        
        diagnosis_query = f"""
        ### DATA
        User Prompt: "{prompt}"
        AI Output: "{output}"
        Judge Reasoning: "{reasoning}"
        Score: {score}/1.0
        
        ### TASK
        Complete this sentence: "The output can be improved if the prompt..."
        Focus on specific actionable changes (Structure, Persona, Constraints, Examples).
        """
        
        try:
            diagnosis_text = self.model_wrapper.generate(
                prompt=diagnosis_query, 
                system_instruction=system_prompt, 
                max_new_tokens=60, 
                temperature=0.3
            ).strip()
            
            clean = diagnosis_text.replace("###", "").strip()
            if not clean.lower().startswith("the output"):
                return f"The output can be improved if the prompt addresses: {clean}"
            return clean
        except:
            return "The output can be improved if the prompt is more specific."

    # ---------------- 3. The Strategist (Aggressive & Stateful) ----------------
    def generate_dynamic_strategies(self, prompt, root_cause_sentence, used_strategies=[]):
        avoid_text = ""
        if used_strategies:
            avoid_list = used_strategies[-5:]
            avoid_text = f"DO NOT use these previous strategies: {'; '.join(avoid_list)}"

        query = f"""
        ### ROLE
        You are a Prompt Engineering Expert.
        
        ### CONTEXT
        User Prompt: "{prompt}"
        Diagnosis: "{root_cause_sentence}"
        {avoid_text}
        
        ### EXAMPLES OF POWERFUL STRATEGIES
        1. "Rewrite prompt to force a Persona: 'Act as a Senior Physicist...'"
        2. "Rewrite prompt to force Structure: 'Use Markdown headers and bullet points...'"
        3. "Rewrite prompt to add Constraints: 'Explain in exactly 3 paragraphs...'"
        
        ### TASK
        Write 2 specific instructions to rewrite the prompt.
        Focus on STRUCTURE and FORMAT changes, not just "more detail".
        
        ### OUTPUT
        Strategy 1: ...
        Strategy 2: ...
        """
        
        try:
            suggestion_text = self.model_wrapper.generate(
                prompt=query, 
                system_instruction="You are a helpful expert.", 
                max_new_tokens=150, 
                temperature=0.85 
            )
            strategies = []
            for line in suggestion_text.split('\n'):
                if "Strategy" in line and ":" in line:
                    strategies.append(line.split(":", 1)[1].strip())
            
            return strategies[:2] if strategies else ["Rewrite to be more clear."]
        except:
            return ["Rewrite to be more direct."]

    # ---------------- 4. The Executor ----------------
    def apply_strategy(self, current_prompt, strategies):
        candidates = []
        base_prompt = self._clean_previous_constraints(current_prompt)

        for instruction in strategies:
            if "append" in instruction.lower() or "add" in instruction.lower():
                match = re.search(r"['\"](.*?)['\"]", instruction)
                if match:
                    to_add = match.group(1)
                    candidates.append(f"{base_prompt}\n\n[INSTRUCTION: {to_add}]")
                    continue
            
            full_task = f"""
            Original Prompt: "{base_prompt}"
            Improvement Instruction: "{instruction}"
            
            Task: Rewrite the prompt to incorporate the instruction. 
            Output ONLY the new prompt. Do not output explanations.
            """
            new_prompt = self.model_wrapper.generate(
                prompt=full_task, 
                system_instruction="You are a Prompt Rewriter. Output only the prompt text.", 
                temperature=0.8, 
                max_new_tokens=300
            ).strip().replace('"', '')
            
            if len(new_prompt) > 5 and not self._is_refusal(new_prompt):
                candidates.append(new_prompt)
                
        return candidates

    def _clean_previous_constraints(self, text):
        pattern = r"\n\n\[INSTRUCTION: .*?\]"
        return re.sub(pattern, "", text, flags=re.DOTALL).strip()

    def _is_refusal(self, text):
        keywords = ["i can't", "i cannot", "unable to", "as an ai", "sorry", "limitations"]
        if len(text.split()) < 40 and any(x in text.lower() for x in keywords):
            return True
        return False

    def _check_repetition(self, text):
        lines = [x for x in text.split('\n') if len(x) > 10]
        if len(lines) > 4:
            if lines[-1] == lines[-2] == lines[-3]: return True
        return False

    # ---------------- MAIN PIPELINE ----------------
    def optimize(self, initial_prompt, num_rounds=7):
        SCORE_THRESHOLD = 0.95
        PATIENCE_LIMIT = 2  # Max rounds to wait without improvement
        
        all_attempted_strategies = []
        all_data = []
        patience_counter = 0 # Tracks stagnation
        
        yield {"type": "progress", "message": "Analyzing Intent..."}
        user_intent = self._extract_intent(initial_prompt)
        yield {"type": "progress", "message": f"Intent Detected: {user_intent}"}

        # Baseline
        yield {"type": "progress", "message": "Establishing Baseline..."}
        best_output = self.model_wrapper.generate(initial_prompt, max_new_tokens=400)
        best_score, best_reasoning = self.compute_llm_score(initial_prompt, best_output, user_intent)
        
        global_best = {
            "prompt": initial_prompt, 
            "score": best_score, 
            "output": best_output, 
            "reasoning": best_reasoning,
            "round": 0
        }
        
        yield {"type": "progress", "message": f"Baseline Score: {best_score} ({best_reasoning[:60]}...)"}

        current_prompt = initial_prompt
        current_output = best_output
        current_score = best_score
        current_reasoning = best_reasoning

        for rnd in range(1, num_rounds + 1):
            if current_score >= SCORE_THRESHOLD:
                yield {"type": "progress", "message": "Excellent score achieved. Early stopping."}
                break

            yield {"type": "progress", "message": f"=== Round {rnd} ==="}
            
            # Diagnose
            root_cause = self.diagnose_root_cause(current_prompt, current_output, current_score, current_reasoning)
            
            # Plan
            strategies = self.generate_dynamic_strategies(current_prompt, root_cause, all_attempted_strategies)
            all_attempted_strategies.extend(strategies)
            
            # Execute
            candidates = self.apply_strategy(current_prompt, strategies)
            
            if not candidates:
                yield {"type": "progress", "message": "No valid strategies found. Retrying..."}
                continue

            round_candidates = []
            for idx, cand_prompt in enumerate(candidates):
                cand_output = self.model_wrapper.generate(cand_prompt, max_new_tokens=400)
                cand_score, cand_reasoning = self.compute_llm_score(cand_prompt, cand_output, user_intent)
                
                strat_name = strategies[idx] if idx < len(strategies) else "Variation"
                strat_short = (strat_name[:45] + '...') if len(strat_name) > 45 else strat_name

                row = {
                    "Round": rnd, 
                    "Prompt No.": idx+1, 
                    "Strategy": strat_short, 
                    "Prompt": cand_prompt, 
                    "Output": cand_output, 
                    "Final Score": cand_score
                }
                all_data.append(row)
                round_candidates.append(row)
                
                if cand_score > global_best["score"]:
                    global_best = {
                        "prompt": cand_prompt, 
                        "score": cand_score, 
                        "output": cand_output,
                        "round": rnd
                    }

            round_candidates.sort(key=lambda x: x["Final Score"], reverse=True)
            
            if round_candidates:
                winner = round_candidates[0]
                yield {"type": "progress", "message": f"Round Best: {winner['Final Score']}"}
                
                # --- PATIENCE LOGIC ---
                if winner["Final Score"] > current_score:
                    # Improvement found! Reset patience.
                    current_prompt = winner["Prompt"]
                    current_output = winner["Output"]
                    current_score = winner["Final Score"]
                    current_reasoning = cand_reasoning 
                    patience_counter = 0 
                else:
                    # Stagnation (Winner <= Current)
                    patience_counter += 1
                    yield {"type": "progress", "message": f"No improvement. Stagnation count: {patience_counter}/{PATIENCE_LIMIT}"}
                    
                    if patience_counter >= PATIENCE_LIMIT:
                        yield {"type": "progress", "message": "Performance stagnated. Stopping early."}
                        break
            
        yield {
            "type": "result", 
            "best_prompt": global_best["prompt"], 
            "best_score": global_best["score"],
            "best_output": global_best["output"], 
            "all_candidates": pd.DataFrame(all_data).fillna(0).to_dict(orient="records")
        }