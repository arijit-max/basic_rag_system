import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_documents_from_directory(directory_path = r"C:\Users\barua\OneDrive\Desktop\GenAI_INTERVIEW_PREP\RAG SYSTEM\docs"):
    """
    Load documents from a specified directory.

    Args:
        directory_path (str): The path to the directory containing documents."""
    print(f"Loading documents from directory: {directory_path}")
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")
    #Loads all .txt files in the directory and its subdirectories
    loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={
        "encoding": "utf-8"
    })
    documents = loader.load()
    #Check if any documents were loaded
    if(len(documents) == 0):
        raise ValueError(f"No documents found in the directory {directory_path}.")
    return documents

def document_splitter(documents, chunk_size=700, chunk_overlap=100):
    """
    Splits the documents into smaller chunks for processing.
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, persist_directory="vector_store"):
    """
    Create and persist a vector store from the document chunks.
    """

    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata = {"hnsw:space": "cosine"} #cosine similarity is used for vector search
    )
    print(f"Vector store created and persisted at {persist_directory}.")
    return vector_store


def main():
    documents = load_documents_from_directory()
    # Split the documents into chunks
    chunks = document_splitter(documents)

    vector_store = create_vector_store(chunks)

if __name__ == "__main__":
    main()



