import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Load website text
with open("data/website.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split into simple chunks
chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_numpy=True)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save index & chunks
faiss.write_index(index, "index/faiss.index")
with open("index/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print("Index built and saved!")
