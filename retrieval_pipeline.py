import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

persist_directory = "vector_store"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
db = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings,
    collection_metadata = {"hnsw:space": "cosine"}) #cosine similarity is used for vector search

query = input("Enter your query: ")

retriever = db.as_retriever(search_kwargs={"k": 3}) #Retrieve top 3 chunks with highest similarity score to the query


# retriever = db.as_retriever(
#     search_kwargs={"k": 3, 
#                    "score_threshold": 0.3}) #Retrieve top 3 chunks with highest similarity score to the query and filter out chunks with similarity score below 0.3

#Take the user's query, search the vector database for the most semantically relevant documents/chunks, and return them.
relevant_docs = retriever.invoke(query)

print("---Context---")
for i, doc in enumerate(relevant_docs):
    print(f"\n--- Document {i + 1} ---")
    print(doc.page_content)
    print(doc.metadata)

context = "\n\n".join(doc.page_content for doc in relevant_docs)
combined_input = f"""Based on the following context, answer the question: {query}

Context:
{context}

Please provide a concise and accurate answer to the question based on the context provided. If the context does not contain enough information to answer the question, please indicate that as well.
"""
client = InferenceClient(
    model="Qwen/Qwen3-8B",
    token=HF_TOKEN
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that provides accurate and concise answers based on the provided context.",
    },
    {"role": "user", "content": combined_input},
]

result = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=messages
)
print("---Answer---")
print(result.choices[0].message.content)
