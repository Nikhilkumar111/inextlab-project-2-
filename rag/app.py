import os
import json
import faiss
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from prompt import build_prompt   # 👈 NEW

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

# Load FAISS index
index = faiss.read_index("index/faiss.index")

# Load chunks
with open("index/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def ask_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

    response = requests.post(
        url,
        params={"key": API_KEY},
        json={
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }
    )

    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def ask_question(question, top_k=3):
    # 1️⃣ Create embedding
    q_emb = embed_model.encode([question], convert_to_numpy=True)

    # 2️⃣ FAISS search
    D, I = index.search(q_emb, top_k)
    context = "\n".join([chunks[i] for i in I[0]])

    # 3️⃣ Build prompt (separated)
    prompt = build_prompt(context, question)

    # 4️⃣ Gemini response
    return ask_gemini(prompt)


if __name__ == "__main__":
    print("Type 'exit' to quit.")
    while True:
        q = input("Ask a question: ").strip()
        if q.lower() == "exit":
            break
        print("Answer:", ask_question(q))
        print("-" * 50)
