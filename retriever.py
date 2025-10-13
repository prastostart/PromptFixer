# retriever.py
class SmartRetriever:
    def __init__(self, docs):
        """
        docs: list of verified finance knowledge sentences
        """
        self.docs = docs

    def retrieve(self, query, top_k=3):
        """
        Returns the top_k most relevant sentences based on simple keyword overlap.
        """
        query_words = set(query.lower().split())
        scored = []
        for doc in self.docs:
            doc_words = set(doc.lower().split())
            score = len(query_words & doc_words)
            scored.append((score, doc))
        scored.sort(reverse=True)
        top_docs = [doc for _, doc in scored[:top_k]]
        return "\n".join(top_docs)

