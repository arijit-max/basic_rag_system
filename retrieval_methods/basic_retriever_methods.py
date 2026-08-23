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

# ========================================================================================
# METHOD 1: Basic similarity search
# Drawbacks --> Will return top k chunks even if there is no relation of question with 
# document provided
# ========================================================================================

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

# ========================================================================================
# METHOD 2: Similarity with Score Threshold
# Retrieves chunks with similarity score higher than threshold value
# But if threshold is set too high say 0.7 it will not fetch any chunk even if
# it matches
# ========================================================================================

print("\n= METHOD 2: Similarity with Score Threshold === ")
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
            "k": 3,
            "score_threshold": 0.3 # Only return docs with similarity >= 0.3
    }   
)

docs = retriever.invoke (query)
print(f"Retrieved {len(docs)} documents (threshold: 0.3):\n")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")

print("-" * 60)

# ========================================================================================
# METHOD 3: Maximum Marginal Relevance
# Balances relevance and diversity - avoid redundant results
# ========================================================================================

retriever = db.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k":3, # Final number of docs
        "fetch_k":10, # Initial pool to select from
        "lambda_mult":0.5 # 0 = max diversity and 1 = max relevance
    }
)
docs = retriever.invoke (query)
print(f"Retrieved {len(docs)} documents:\n")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")

print("-" * 60)





