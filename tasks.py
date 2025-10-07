from sklearn.metrics import accuracy_score

class ExampleTask:
    def __init__(self):
        # toy dataset
        self.data = [
            ("I love this movie", "positive"),
            ("This is the worst day", "negative"),
            ("Absolutely fantastic!", "positive"),
            ("Terrible and boring", "negative"),
        ]

    def evaluate(self, model, prompt):
        preds, labels = [], []
        for text, label in self.data:
            out = model.generate(prompt, text).lower()
            pred = "positive" if "positive" in out else "negative"
            preds.append(pred)
            labels.append(label)

        # Here you could compute ACR/BCR/regression instead
        return accuracy_score(labels, preds)

