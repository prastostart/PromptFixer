import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

class TextModel:
    def __init__(self, model_name=None):
        # Configuration for the Qwen 7B GGUF Model
        repo_id = "bartowski/Qwen2.5-7B-Instruct-GGUF"
        filename = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        
        print(f"Initializing Qwen 2.5 7B (Quantized)...")
        model_path = f"./models_cache/{filename}"
        
        # Auto-download if it doesn't exist
        if not os.path.exists(model_path):
            print("Model not found locally. Downloading...")
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id, 
                    filename=filename, 
                    local_dir="./models_cache"
                )
            except Exception as e:
                print(f"CRITICAL ERROR: Could not download model. {e}")
                raise e

        # --- KEY FIXES HERE ---
        # n_ctx=8192: Increases the "memory" of the conversation. 
        #             The default is too small for the Judge prompts.
        # n_batch=512: Optimizes processing speed on Apple Silicon.
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,      # Offload all to Metal (GPU)
            n_ctx=8192,           # INCREASED from 4096 (Prevents -1 error)
            n_batch=512,          # Batch size for prompt processing
            verbose=False         # Set to True if you want to see debug stats
        )
        print("Model loaded on Apple Silicon (Metal) successfully.")

    def generate(self, prompt, max_new_tokens=200, temperature=0.7, system_instruction=None):
        messages = []
        
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        messages.append({"role": "user", "content": prompt})

        try:
            # Create completion
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            return output['choices'][0]['message']['content'].strip()
            
        except ValueError as e:
            # This catches context limit errors specifically
            if "exceeds context" in str(e) or "llama_decode" in str(e):
                print(f"ERROR: Context Limit Hit. (Prompt len: {len(prompt)})")
                return "[Error: Input too long for context window]"
            print(f"Generation Error: {e}")
            return "Error generating response."
        except Exception as e:
            print(f"CRITICAL Generation Error: {e}")
            return "Error."