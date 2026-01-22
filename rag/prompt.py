def build_prompt(context, question):
    return f"""
You are a helpful assistant.
Answer the question strictly using the given context.

Context:
{context}

Question:
{question}

Answer:
"""
