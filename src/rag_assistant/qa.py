import json
import sys
from typing import List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import settings


def retrieve(query: str, k: int = 4) -> List[Dict]:
    """Return top-k docs with source and content."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    vs = Chroma(embedding_function=embeddings, persist_directory=settings.CHROMA_DIR)
    docs = vs.similarity_search(query, k=k)
    return [
        {
            "source": d.metadata.get("source", ""),
            "content": d.page_content,
        }
        for d in docs
    ]


def answer_extractive(query: str, k: int = 4, max_chars: int = 1500) -> Dict:
    """Simple extractive answer: stitch top-k chunks + return unique sources."""
    hits = retrieve(query, k=k)
    if not hits:
        return {"answer": "No results found. Did you run ingestion?", "sources": []}

    stitched = "\n---\n".join(h["content"] for h in hits)
    # Keep answer concise for CLI display
    answer = stitched[:max_chars]
    sources = list({h["source"] for h in hits if h["source"]})

    return {"answer": answer, "sources": sources}


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m rag_assistant.qa "Your question"')
        sys.exit(1)

    # Allow spaces in the question (join all args)
    query = " ".join(sys.argv[1:])
    result = answer_extractive(query)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
