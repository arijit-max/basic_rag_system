import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from openai import OpenAI
import json
from pydantic import BaseModel
from typing import List 

load_dotenv()

persist_directory = "vector_store"

OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
GENERATION_MODEL = "mimo-v2.5"
OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
db = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings,
    collection_metadata = {"hnsw:space": "cosine"}) #cosine similarity is used for vector search

class QueryVariations(BaseModel): # Pydantic models for structured o/p
    queries: List[str]

query = input("Enter your query: ")

# ========================================================================================
# Step 1: Generate multiple query variations
# ========================================================================================

if not OPENCODE_API_KEY:
    raise ValueError("OPENCODE_API_KEY is not set in the environment")

client = OpenAI(api_key=OPENCODE_API_KEY, base_url=OPENCODE_BASE_URL)
prompt = f"""Generate three different variations of this query that would help to get relevant documents.
ORIGINAL QUERY: {query}
Return valid JSON in exactly this format: {{"queries": ["variation 1", "variation 2", "variation 3"]}}
Each variation should rephrase or approach the same question from a different angle.
"""
response = client.chat.completions.create(
    model=GENERATION_MODEL,
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
    temperature=0.2,
    max_tokens=300,
)
query_variations = QueryVariations.model_validate(
    json.loads(response.choices[0].message.content)
).queries
print("Generate queries..")
for i,variation in enumerate(query_variations,1):
    print(f"{i}. {variation}")

# ========================================================================================
# Step 2: Search with each query variations and store result
# ========================================================================================

retriever = db.as_retriever(
    search_kwargs = {
        "k":5
    }
)
all_retrieval_results = []
for i,query in enumerate (query_variations,1):
    print(f"Results for {i}: {query} == ")

    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)

    print(f"Retrieved {len(docs)} documents\n")

    for j,doc in enumerate(docs,1):
        print(f"Document: {j}:")
        print(f"{doc.page_content[:150]}...\n")
    print("-" * 50)
print("-" * 60)
print("Multi Query retrieval complete!")

# ========================================================================================
# Step 3: Apply Reciprocal Rank Fusion (RRF)
# ========================================================================================

def reciprocal_rank_fusion(chunk_list,k = 60, verbose = True): 
    #Show me detailed information about what is happening internally while this code runs
    if verbose:
        print("\n"+"-" * 60)
        print("Applying RRF")
        print("-" * 60)
        print("Calculating RRF Scores...")

    # rrf_score = defaultdict(float) # will store {chunk_content: rrf_score}
    all_unique_chunks = {} # stores {chunk_content: actual_chunk_object}














