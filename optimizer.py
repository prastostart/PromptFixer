from models import TextModel
from retriever import LocalRetriever
import pandas as pd
from sentence_transformers import SentenceTransformer
import re

class PromptOptimizer:
    def __init__(self):
        # Initialize the Model (Qwen 7B)
        self.model_wrapper = TextModel()
        
        # Initialize Embeddings & Retriever
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.retriever = LocalRetriever(embedding_model=self.embedding_model)

    # ---------------- 0. Intent Classifier ----------------
    def _extract_intent(self, prompt):
        try:
            intent = self.model_wrapper.generate(
                prompt=f"User Prompt: '{prompt}'\nTask: Identify intent in 3-5 words.\nOutput:", 
                max_new_tokens=20, 
                temperature=0.1
            ).strip()
            return intent
        except: 
            return "General query"

    # ---------------- 1. The Judge (RAG Aware + Un-Anchored) ----------------
    def compute_llm_score(self, prompt, output, original_intent, has_context=False):
        # 1. Hygiene Check
        if self._is_refusal(output): 
            return 0.0, "Refusal detected."
        
        redundancy_penalty = 1.0
        if self._check_repetition(output): 
            redundancy_penalty = 0.5

        # 2. RAG Specific Instructions
        rag_instruction = ""
        if has_context:
            rag_instruction = """
            CRITICAL RAG CHECK: 
            - The User provided CONTEXT (Source Material).
            - The Output MUST be based on this context.
            - If the Output uses outside knowledge NOT in the context, Score must be < 0.5.
            - If the Output ignores the context completely, Score must be 0.0.
            """

        judge_query = f"""
        ### CONTEXT
        User Intent: "{original_intent}"
        Current Prompt: "{prompt}"
        AI Output: "{output}"
        
        ### SCORING RUBRIC
        {rag_instruction}
        - 0.0: Hallucination (if RAG), Refusal, or Wrong.
        - 0.5: Correct answer, but basic/boring. (PASSING)
        - 0.6: Slightly better structure or detail. (IMPROVED)
        - 0.7: Good, grounded, clear structure. (GOOD)
        - 0.8: Very strong, excellent formatting. (GREAT)
        - 1.0: Perfect.
        
        ### TASK
        Evaluate the AI Output.
        If it is "Correct" but could be structured better, give 0.5 or 0.6.
        If it is "Good", give 0.7.
        
        ### RESPONSE FORMAT
        Reasoning: [1 sentence analysis]
        Score: [0.0-1.0]
        """
        
        try:
            eval_text = self.model_wrapper.generate(
                prompt=judge_query, 
                max_new_tokens=150, 
                temperature=0.1
            ).strip()
            
            # Robust Regex to extract score
            match = re.search(r"Score:\s*(0\.\d+|1\.0|0|1)", eval_text, re.IGNORECASE)
            raw_score = float(match.group(1)) if match else 0.5
            
            reasoning = eval_text.split("Score:")[0].replace("Reasoning:", "").strip()
        except:
            raw_score = 0.5
            reasoning = "Error in evaluation."

        return round(raw_score * redundancy_penalty, 4), reasoning

    # ---------------- 2. The Analyst ----------------
    def diagnose_root_cause(self, prompt, output, score, reasoning):
        if score > 0.95: return "The output is excellent."
        
        diagnosis_query = f"""
        ### DATA
        Prompt: "{prompt}"
        Output: "{output}"
        Judge Reasoning: "{reasoning}"
        Score: {score}/1.0
        
        ### TASK
        Complete sentence: "The output can be improved if the prompt..."
        Focus on actionable changes (Structure, Constraints, Grounding).
        """
        try:
            text = self.model_wrapper.generate(prompt=diagnosis_query, max_new_tokens=60, temperature=0.3).strip()
            clean = text.replace("###", "").strip()
            if not clean.lower().startswith("the output"):
                return f"The output can be improved if the prompt addresses: {clean}"
            return clean
        except:
            return "The output can be improved if the prompt is more specific."

    # ---------------- 3. The Strategist (Memory + Aggressive) ----------------
    def generate_dynamic_strategies(self, prompt, root_cause, used_strategies=[], past_experience=None):
        # Avoid loops
        avoid_text = f"DO NOT use: {'; '.join(used_strategies[-5:])}" if used_strategies else ""
        
        # Inject Memory
        memory_text = ""
        if past_experience:
            memory_text = f"""
            ### MEMORY RECALL
            We have successfully solved a similar prompt ("{past_experience['similar_to'][:30]}...") before.
            The winning strategy was: "{past_experience['strategy']}".
            STRONGLY CONSIDER ADAPTING THIS STRATEGY.
            """

        query = f"""
        Diagnosis: "{root_cause}"
        {avoid_text}
        {memory_text}
        
        Task: Write 2 specific instructions to rewrite the prompt.
        Focus on STRUCTURE, CONSTRAINTS, and FORMATTING.
        
        Output:
        Strategy 1: ...
        Strategy 2: ...
        """
        try:
            txt = self.model_wrapper.generate(prompt=query, max_new_tokens=200, temperature=0.85)
            strategies = [line.split(":", 1)[1].strip() for line in txt.split('\n') if "Strategy" in line and ":" in line]
            return strategies[:2] if strategies else ["Make it more specific."]
        except: return ["Rewrite to be clearer."]

    # ---------------- 4. The Executor ----------------
    def apply_strategy(self, current_prompt, strategies):
        candidates = []
        base_prompt = self._clean_previous_constraints(current_prompt)

        for instruction in strategies:
            # Direct Append
            if "append" in instruction.lower() or "add" in instruction.lower():
                match = re.search(r"['\"](.*?)['\"]", instruction)
                if match:
                    to_add = match.group(1)
                    candidates.append(f"{base_prompt}\n\n[INSTRUCTION: {to_add}]")
                    continue
            
            # Generative Rewrite
            full_task = f"Original: {base_prompt}\nInstruction: {instruction}\nTask: Rewrite prompt. Output ONLY prompt."
            new_prompt = self.model_wrapper.generate(prompt=full_task, temperature=0.8, max_new_tokens=300).strip().replace('"', '')
            if len(new_prompt) > 5 and not self._is_refusal(new_prompt):
                candidates.append(new_prompt)
        return candidates

    def _clean_previous_constraints(self, text):
        return re.sub(r"\n\n\[INSTRUCTION: .*?\]", "", text, flags=re.DOTALL).strip()

    def _is_refusal(self, text):
        return len(text.split()) < 40 and any(x in text.lower() for x in ["i can't", "i cannot", "sorry"])

    def _check_repetition(self, text):
        lines = [x for x in text.split('\n') if len(x) > 10]
        return len(lines) > 4 and lines[-1] == lines[-2] == lines[-3]

    # ---------------- MAIN PIPELINE ----------------
    def optimize(self, initial_prompt, num_rounds=7, use_rag=False):
        SCORE_THRESHOLD = 0.95
        PATIENCE_LIMIT = 2
        
        # 0. Immediate Yield (Prevents frontend "No Output" hang)
        yield {"type": "progress", "message": "Initializing Optimizer..."}

        # 1. RAG Retrieval (Safe Mode)
        retrieved_context = ""
        if use_rag:
            try:
                yield {"type": "progress", "message": "Retrieving Context..."}
                retrieved_context = self.retriever.query_docs(initial_prompt)
                if retrieved_context:
                    yield {"type": "progress", "message": f"Context Found ({len(retrieved_context)} chars)."}
                else:
                    yield {"type": "progress", "message": "No relevant documents found in DB."}
            except Exception as e:
                yield {"type": "progress", "message": f"⚠️ RAG Error (Skipping): {str(e)}"}

        # 2. Memory Retrieval (Safe Mode)
        past_experience = None
        try:
            past_experience = self.retriever.retrieve_experience(initial_prompt)
            if past_experience:
                 yield {"type": "progress", "message": f"🧠 Memory Recall: Found strategy from '{past_experience['similar_to'][:20]}...'"}
        except Exception as e:
            # Non-critical, just log and continue
            print(f"Memory Error: {e}")

        # Helper to construct Prompt + Context
        def construct_input(prompt_text):
            if retrieved_context:
                return f"{prompt_text}\n\n### CONTEXT (Use this strictly):\n{retrieved_context}"
            return prompt_text

        all_attempted_strategies = []
        all_data = []
        patience_counter = 0
        
        yield {"type": "progress", "message": "Analyzing Intent..."}
        user_intent = self._extract_intent(initial_prompt)
        
        # Baseline
        yield {"type": "progress", "message": "Establishing Baseline..."}
        full_input_baseline = construct_input(initial_prompt)
        best_output = self.model_wrapper.generate(full_input_baseline, max_new_tokens=400)
        baseline_score, best_reasoning = self.compute_llm_score(initial_prompt, best_output, user_intent, has_context=bool(retrieved_context))
        
        global_best = {
            "prompt": initial_prompt, 
            "score": baseline_score, 
            "output": best_output, 
            "round": 0,
            "winning_strategy": "None (Baseline)"
        }
        
        yield {"type": "progress", "message": f"Baseline Score: {baseline_score}"}

        current_prompt = initial_prompt
        current_output = best_output
        current_score = baseline_score
        current_reasoning = best_reasoning

        for rnd in range(1, num_rounds + 1):
            if current_score >= SCORE_THRESHOLD:
                yield {"type": "progress", "message": "Success. Stopping."}; break

            yield {"type": "progress", "message": f"=== Round {rnd} ==="}
            
            root_cause = self.diagnose_root_cause(current_prompt, current_output, current_score, current_reasoning)
            
            # Pass Memory + History to Strategist
            strategies = self.generate_dynamic_strategies(current_prompt, root_cause, all_attempted_strategies, past_experience)
            all_attempted_strategies.extend(strategies)
            
            candidates = self.apply_strategy(current_prompt, strategies)
            if not candidates: 
                yield {"type": "progress", "message": "No strategies generated. Retrying..."}
                continue

            round_candidates = []
            for idx, cand_prompt in enumerate(candidates):
                full_input = construct_input(cand_prompt)
                cand_output = self.model_wrapper.generate(full_input, max_new_tokens=400)
                cand_score, cand_reasoning = self.compute_llm_score(cand_prompt, cand_output, user_intent, has_context=bool(retrieved_context))
                
                strat = strategies[idx] if idx < len(strategies) else "Variation"
                row = {"Round": rnd, "Prompt No.": idx+1, "Strategy": strat[:40], "Prompt": cand_prompt, "Output": cand_output, "Final Score": cand_score}
                all_data.append(row); round_candidates.append(row)
                
                if cand_score > global_best["score"]:
                    global_best = {
                        "prompt": cand_prompt, "score": cand_score, "output": cand_output, "round": rnd,
                        "winning_strategy": strat
                    }

            round_candidates.sort(key=lambda x: x["Final Score"], reverse=True)
            
            if round_candidates:
                winner = round_candidates[0]
                yield {"type": "progress", "message": f"Round Best: {winner['Final Score']}"}
                
                if winner["Final Score"] > current_score:
                    current_prompt = winner["Prompt"]
                    current_output = winner["Output"]
                    current_score = winner["Final Score"]
                    current_reasoning = cand_reasoning
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE_LIMIT:
                        yield {"type": "progress", "message": "Stagnation detected. Stopping early."}
                        break
        
        # 3. Save Memory (If successful)
        improvement = global_best["score"] - baseline_score
        if improvement >= 0.1 or global_best["score"] > 0.8:
            if global_best["winning_strategy"] != "None (Baseline)":
                try:
                    yield {"type": "progress", "message": "💾 Saving successful strategy to Memory..."}
                    self.retriever.store_experience(
                        user_prompt=initial_prompt, 
                        winning_strategy=global_best["winning_strategy"],
                        score_improvement=improvement
                    )
                except Exception as e:
                    print(f"Memory Save Failed: {e}")

        yield {
            "type": "result", 
            "best_prompt": global_best["prompt"], 
            "best_score": global_best["score"], 
            "best_output": global_best["output"], 
            "all_candidates": pd.DataFrame(all_data).fillna(0).to_dict(orient="records")
        }