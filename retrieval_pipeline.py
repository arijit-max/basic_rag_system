from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

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






