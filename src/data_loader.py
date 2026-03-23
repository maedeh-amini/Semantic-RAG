

from pathlib import Path
from typing import List, Any
from datasets import load_dataset
from langchain_core.documents import Document

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load Hugging Face RAGBench HotpotQA train split, extract 'documents_sentences',
    flatten nested sentences, and convert to LangChain Document objects.
    """
    documents = []

    # ------------------------
    # 1. Load Hugging Face dataset
    # ------------------------
    try:
        # Load only 'hotpotqa' config
        dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="train")
        print(f"[DEBUG] Loaded Hugging Face HotpotQA train split with {len(dataset)} rows")

        # Select only 'documents_sentences' column
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != "documents_sentences"])
        
        # Flatten nested lists: row -> doc -> sentence
        all_sentences = []
        for row in dataset:
            for doc in row["documents_sentences"]:          # outer list of docs
                for sentence in doc:                        # inner list of sentences
                    sent_id, sent_text = sentence
                    all_sentences.append(
                        Document(page_content=sent_text)
                    )

        print(f"[DEBUG] Extracted {len(all_sentences)} sentences as individual Documents")
        documents.extend(all_sentences)

    except Exception as e:
        print(f"[ERROR] Failed to load Hugging Face dataset: {e}")

    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)
    
