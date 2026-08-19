from unittest import result

from dotenv import load_dotenv
from httpx2 import query
from langchain import messages
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceHub
from huggingface_hub import InferenceClient
from langchain_core.messages import SystemMessage, HumanMessage
import os

from opentelemetry import context

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

persist_directory = "vector_store"
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
    )
db = Chroma(persist_directory=persist_directory,embedding_function=embeddings,collection_metadata = {"hnsw:space": "cosine"}) #cosine similarity is used for vector search

client = InferenceClient(
    model="Qwen/Qwen3-8B",
    token=HF_TOKEN
)

chat_history = []

def ask_question(user_query):
    print(f"User query: {user_query}")

    if chat_history:
        messages = [
            SystemMessage(
                content = "Given the chat history, rewrite the new question to be a standalone question. Just provide the rewritten question without any additional commentary."),
        ] + chat_history + [
            HumanMessage(content=user_query)    

        ]
        result = client.invoke(messages)
        search_query = result.content.strip()
        print(f"Rewritten query: {search_query}")
    else:
        search_query = user_query

    retriever = db.as_retriever(search_kwargs={"k": 3}) #Retrieve top 3 chunks with highest similarity score to the query
    relevant_docs = retriever.invoke(search_query)
    combined_input = f"""Based on the following context, answer the question: {query}

    Context:
    {context}

    Please provide a concise and accurate answer to the question based on the context provided. If the context does not contain enough information to answer the question, please indicate that as well.
    """

    messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that provides accurate and concise answers based on the provided context and the chat history.",
    },
    {"role": "user", "content": combined_input},
    ]

    result = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
    messages=messages
    )
    print("---Answer---")
    print(result.choices[0].message.content)

    

def start_chat():
    print("Welcome to the History-Aware Question Answering System!")

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")

        if question.lower() == 'exit':
            print("Exiting the chat. Goodbye!")
            break
        ask_question(question)

if __name__ == "__main__":
    start_chat()





