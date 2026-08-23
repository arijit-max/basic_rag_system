import json
import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

load_dotenv()

OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
PERSIST_DIRECTORY = "multimodalrag_vector"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GENERATION_MODEL = "mimo-v2.5"
OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"


def create_retriever(
    persist_directory: str = PERSIST_DIRECTORY,
    k: int = 3,
):
    """Load the multimodal Chroma store and return a similarity retriever."""
    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(
            f"Vector store directory does not exist: {persist_directory}"
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    multimodalrag_vector = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return multimodalrag_vector.as_retriever(search_kwargs={"k": k})


def retrieve_documents(query: str, retriever, k: int = 3):
    """Return the most relevant multimodal chunks for a query."""
    if not query.strip():
        raise ValueError("Query cannot be empty")

    retriever.search_kwargs["k"] = k
    return retriever.invoke(query)


def get_original_content(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Read the raw multimodal payload saved by multi_modal_rag.py."""
    original_content = metadata.get("original_content", "{}")
    if isinstance(original_content, str):
        try:
            original_content = json.loads(original_content)
        except json.JSONDecodeError:
            original_content = {"raw_text": original_content}

    return {
        "raw_text": original_content.get("raw_text", ""),
        "tables_html": original_content.get("tables_html", []),
        "images_base64": original_content.get("images_base64", []),
    }


def build_context(documents) -> Tuple[str, List[str]]:
    """Build text context and collect images from retrieved chunks."""
    context_parts = []
    images = []

    for index, document in enumerate(documents, start=1):
        content = get_original_content(document.metadata)
        text = content["raw_text"] or document.page_content
        context_parts.append(f"Document {index}:\n{text}")

        for table in content["tables_html"]:
            context_parts.append(f"Table from document {index}:\n{table}")
        images.extend(content["images_base64"])

    return "\n\n".join(context_parts), images


def answer_query(query: str, documents) -> str:
    """Generate a grounded answer with OpenCode's multimodal model."""
    if not OPENCODE_API_KEY:
        raise ValueError("OPENCODE_API_KEY is not set in the environment")

    context, images = build_context(documents)
    prompt = f"""Answer the user's question using only the retrieved context below.
If the context does not contain enough information, say so clearly.

User question:
{query}

Retrieved context:
{context}

Give a concise, accurate answer and mention relevant figures or table values when present."""

    message_content = [{"type": "text", "text": prompt}]
    for image_base64 in images:
        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            }
        )
    client = OpenAI(
        api_key=OPENCODE_API_KEY,
        base_url=OPENCODE_BASE_URL,
    )
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": message_content}],
        temperature=0.2,
        max_tokens=1000,
    )
    message = response.choices[0].message
    return message.content or "No answer was generated."


def main() -> None:
    query = input("Enter your query: ").strip()
    retriever = create_retriever()
    relevant_documents = retrieve_documents(query, retriever)

    print("\n--- Retrieved context ---")
    for index, document in enumerate(relevant_documents, start=1):
        print(f"\n--- Document {index} ---\n{document.page_content}")

    print("\n--- Answer ---")
    print(answer_query(query, relevant_documents))


if __name__ == "__main__":
    main()
