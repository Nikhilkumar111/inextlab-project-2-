# RAG Application using FAISS and Gemini API

## 📌 Project Overview
This project implements a **small Retrieval-Augmented Generation (RAG) application** using **FAISS** and **iNextLabs website data**, built **without using any frameworks**.

The application retrieves relevant information from indexed iNextLabs content and generates accurate, grounded answers using the **Google Gemini (gemini-2.5-flash)** model.

---

## 🧠 What is RAG?
Retrieval-Augmented Generation (RAG) improves language model responses by:
1. Retrieving relevant information from a vector database
2. Injecting the retrieved context into the model prompt
3. Generating answers grounded in retrieved data rather than memory

This approach reduces hallucinations and improves factual accuracy.

---

## 🏗️ Project Architecture

1. **Data Source**  
   iNextLabs website content is stored as text and used as the knowledge base.

2. **Text Chunking**  
   The data is split into small, meaningful chunks to preserve semantic context.

3. **Embedding Generation**  
   Each chunk is converted into vector embeddings using a Sentence Transformer model.

4. **Vector Storage (FAISS)**  
   Embeddings are stored in a FAISS index to enable fast similarity search.

5. **Retrieval**  
   User queries are embedded and compared with stored vectors to retrieve the most relevant chunks.

6. **Answer Generation**  
   Retrieved context is passed to the Gemini API to generate a grounded response.

---

## 📂 Project Structure

rag/
│
├── app.py # Main RAG application
├── build_index.py # Builds FAISS index from website data
├── requirements.txt # Python dependencies
├── .env # Gemini API key
│
├── data/
│ └── website.txt # iNextLabs website content
│
├── index/
│ ├── faiss.index # FAISS vector index
│ └── chunks.json # Stored text chunks
│
└── README.md