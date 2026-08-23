import json
from typing import List
# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured. chunking. title import chunk_by_title
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
FILE_PATH = os.getenv("FILE_PATH")
OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
GENERATION_MODEL = "mimo-v2.5"

# ==============================================================================
# Step 1: Load the PDF and partition it into chunks by unstructured library 
# ==============================================================================

def partition_document(file_path:str):
    """
    Extracts elements from pdf 
    """
    print("Partitioning document...")
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res", #Uses most accurate OCR strategy for text extraction but slower
        infer_table_structure=True, # Keeps table structure as HTML not jumbled texts
        extract_image_block_types = ["Image"], # Grabs images from the document
        extract_image_block_to_payload=True # Stores image as base64 in the payload of the document
    )
    print(f"Partitioned {len(elements)} elements from the document.")
    return elements

# ==============================================================================
# Step 2: Chunk the document by title using unstructured library
# ==============================================================================

def create_chunks_by_title(elements):
    """
    Chunks the document by title
    """
    print("Chunking document by title...")
    chunks = chunk_by_title(
       elements,
       max_characters=3000,  # Maximum number of characters per chunk
       new_after_n_chars=2400,  # Create a new chunk after this many characters
       combine_text_under_n_chars=500,  # Combine text under this many characters into the previous chunk
    )
    print(f"Created {len(chunks)} chunks from the document.")
    return chunks

# ==============================================================================
# Step 3: Process all chunks with AI summaries
# ==============================================================================

def separate_content_types(chunk):
    """
        Analyze the type of content in a chunk
    """
    content_data = {
        'text': chunk.text,
        'tables':[],
        'images':[],
        'types':['text']
    }

    # Check for tables and images in original document
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata,'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # Handling tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)

            # Handling images
            elif element_type == 'Image':
                if hasattr(element,'metadata') and hasattr(element.metadata,'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text:str, tables: List[str], images: List[str]) -> str:
    """
        Create AI enhanced summary for mixed content
    """
    try:
        if not OPENCODE_API_KEY:
            raise ValueError("OPENCODE_API_KEY is not set in the environment")

        client = OpenAI(
            api_key=OPENCODE_API_KEY,
            base_url=OPENCODE_BASE_URL,
        )
        prompt_text = f"""
        You are creating a searchable description for document content retrieval.
        CONTENT TO ANALYZE:
        TEXT CONTENT: {text}
        """
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        prompt_text += """
        YOUR TASK:
        Generate a comprehensive, searchable description that covers:
        1. Key facts, numbers and data points from text and tables
        2. Main topics and concepts discussed
        3. Questions this content could answer
        4. Visual content analysis (charts, diagrams, and patterns in images)
        5. Alternative search terms users might use

        Make it detailed and searchable - prioritize findability over brevity.
        Return only the searchable description.
        """
        message_content = [{"type": "text", "text": prompt_text}]

        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": message_content}],
            temperature=0.2,
            max_tokens=1200,
        )

        answer = response.choices[0].message.content
        return answer or text
    except Exception as e:
        print(f"AI summary generation failed: {e}")
        return text
        
def summarise_chunks(chunks):
    """
    Summarizes each chunk using AI model
    """
    print("Summarizing chunks...")

    langchain_document = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i+1
        print(f"Processing chunk {current_chunk}/{total_chunks}...")

        content_data = separate_content_types(chunk) # Analyse the chunk content

        print(f"Types found: {content_data['types']}")
        print(f"Tabels found: {content_data['tables']} and Images found: {content_data['images']}")

        # Creating AI summaries for tables and images in a chunk

        if content_data['tables'] or content_data['images']:
            print("Creating AI Summary for mixed content")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'],
                    content_data['images']
                )
                print("AI summary created successfully")
            except Exception as e:
                print(f"AI summary failed: {e}")
                enhanced_content = content_data['text']
        else:
            print(f"Using raw text (no tables/images)")
            enhanced_content = content_data['text']

        # Creating Langchain document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata = {
                "original_content":json.dumps({
                    "raw_text":content_data['text'],
                    "tables_html":content_data['tables'],
                    "images_base64":content_data['images']
                })
            }
        )
        langchain_document.append(doc)
        print(f"Processed {len(langchain_document)} chunks")
    return langchain_document

# ==============================================================================
# Step 4: Storing chunks in vector DB
# ==============================================================================

def create_vector_store(documents, persist_directory="multimodalrag_vector"):
    """Create and persist ChromaDB vector store"""
    print(" Creating embeddings and storing in ChromaDB ... ")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
        )
    # Create ChromaDB vector store
    print(" --- Creating vector store --")
    vectorstore = Chroma. from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"}
    )
    print(" --- Finished creating vector store --- ")

    print(f" Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    """Run multimodal PDF ingestion and create the persistent vector store."""
    if not FILE_PATH:
        raise SystemExit("FILE_PATH is not set in the .env file")
    if not os.path.isfile(FILE_PATH):
        raise FileNotFoundError(f"PDF file does not exist: {FILE_PATH}")

    elements = partition_document(FILE_PATH)
    chunks = create_chunks_by_title(elements)
    documents = summarise_chunks(chunks)
    create_vector_store(documents)


if __name__ == "__main__":
    main()
