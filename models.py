import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TextModel:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        print(f"Loading model: {model_name}")

        # Detect hardware
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16, # Llama 3 prefers bfloat16
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    def generate(self, prompt, max_new_tokens=150, temperature=0.7, system_instruction=None):
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        messages.append({"role": "user", "content": prompt})

        # Apply Chat Template
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()