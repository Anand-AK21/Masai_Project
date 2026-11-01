"""Part D (tasks 1-3): chunking, TF-IDF retrieval and the grounded policy assistant."""
import glob, os, sys
sys.path.insert(0, os.getcwd())
import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_DIR = os.path.join("data", "policies")

def load_docs(folder=POLICY_DIR):
    docs = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        docs[os.path.basename(path)] = open(path, encoding="utf-8").read()
    return docs

def chunk_words(text, size=40, overlap=10):
    """Windows of `size` words, each starting size-overlap words after the previous one.
    Stops after the first window whose end reaches or passes the last word."""
    words, step, chunks, start = text.split(), size - overlap, [], 0
    while True:
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks

class PolicyIndex:
    def __init__(self, size=40, overlap=10, folder=POLICY_DIR):
        self.chunks = [{"source": src, "text": c} for src, t in load_docs(folder).items()
                       for c in chunk_words(t, size, overlap)]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4), "text": self.chunks[i]["text"]} for i in top]

SYSTEM = ("You are a POLICY ASSISTANT for Instamart support. Answer only from the supplied CONTEXT. "
          "If the context does not contain the answer, say that you could not find the information in the policy documents.")

def answer(question, index, k=3):
    hits = index.retrieve(question, k)
    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]
