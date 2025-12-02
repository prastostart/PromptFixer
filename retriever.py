import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import os

class LocalRetriever:
    def __init__(self, embedding_model=None):
        if embedding_model:
            self.encoder = embedding_model
        else:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            
        # Initialize persistent local DB
        self.client = chromadb.PersistentClient(path="./rag_db")
        
        # Collection 1: RAG Documents (Context)
        self.doc_collection = self.client.get_or_create_collection(name="user_docs")
        
        # Collection 2: Optimization History (Experience)
        self.mem_collection = self.client.get_or_create_collection(name="prompt_experience")

    # --- RAG METHODS ---
    def add_document(self, text, filename="doc"):
        chunk_size = 500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        ids = [f"{filename}_{str(uuid.uuid4())[:8]}" for _ in chunks]
        metadatas = [{"source": filename} for _ in chunks]
        embeddings = self.encoder.encode(chunks).tolist()
        
        self.doc_collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
        return len(chunks)

    def query_docs(self, prompt, n_results=3):
        if self.doc_collection.count() == 0: return ""
        query_emb = self.encoder.encode([prompt]).tolist()
        results = self.doc_collection.query(query_embeddings=query_emb, n_results=n_results)
        
        # Check if we actually found anything
        if not results['documents'] or not results['documents'][0]:
            return ""
            
        return "\n\n".join(results['documents'][0])

    def clear_docs(self):
        try:
            self.client.delete_collection("user_docs")
            self.doc_collection = self.client.get_or_create_collection(name="user_docs")
            # Also clear memory to start fresh for testing
            self.client.delete_collection("prompt_experience")
            self.mem_collection = self.client.get_or_create_collection(name="prompt_experience")
            print("[Retriever] Database completely reset.")
        except Exception as e:
            print(f"[Retriever] Error clearing DB: {e}")

    # --- MEMORY METHODS (FIXED) ---
    def store_experience(self, user_prompt, winning_strategy, score_improvement):
        """
        Saves a successful optimization episode.
        """
        doc_id = f"exp_{str(uuid.uuid4())[:8]}"
        embedding = self.encoder.encode([user_prompt]).tolist()
        
        self.mem_collection.add(
            ids=[doc_id],
            embeddings=embedding,
            documents=[winning_strategy],
            metadatas=[{
                "original_prompt": user_prompt, 
                "improvement": float(score_improvement)
            }]
        )
        print(f"[Memory] SAVED strategy: '{winning_strategy[:30]}...' for prompt: '{user_prompt[:30]}...'")

    def retrieve_experience(self, current_prompt, n_results=1):
        """
        Finds if we have successfully optimized similar prompts before.
        """
        if self.mem_collection.count() == 0: 
            return None
        
        query_emb = self.encoder.encode([current_prompt]).tolist()
        
        results = self.mem_collection.query(
            query_embeddings=query_emb, 
            n_results=n_results
        )
        
        # Check results
        if results['documents'] and results['documents'][0]:
            found_dist = results['distances'][0][0]
            found_prompt = results['metadatas'][0][0]['original_prompt']
            
            # DEBUGGING: Look at your terminal to see this
            print(f"[Memory DEBUG] Current: '{current_prompt}'")
            print(f"[Memory DEBUG] Match:   '{found_prompt}' (Dist: {found_dist:.4f})")
            
            # RELAXED THRESHOLD: Changed from 1.0 to 1.5
            # Lower distance = More similar. 
            if found_dist < 1.5: 
                best_strategy = results['documents'][0][0]
                return {"strategy": best_strategy, "similar_to": found_prompt}
            else:
                print(f"[Memory DEBUG] Match ignored (Distance > 1.5)")
        
        return None