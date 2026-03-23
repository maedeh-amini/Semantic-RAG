

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


#------------------------
# Previous Version
# ------------------------

# from pathlib import Path
# from typing import List, Any
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
# from langchain_community.document_loaders import Docx2txtLoader
# from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

# # Hugging Face imports
# from langchain_community.document_loaders import HuggingFaceDatasetLoader
# from datasets import load_dataset, concatenate_datasets
# from langchain_core.documents import Document


# def load_all_documents(data_dir: str) -> List[Any]:
#     """
#     Load all supported local files from the data directory as well as Hugging Face dataset and convert them to LangChain document structure.
#     Local File Support: PDF, TXT, CSV, Excel, Word, JSON
#     HF Dataset support: galileo-ai/ragbench
#     """


#     # Use project root data folder
#     data_path = Path(data_dir).resolve()
#     print(f"[DEBUG] Data path: {data_path}")

#     documents = []


#     # ------------------------
#     # 1. Load Hugging Face dataset
#     # ------------------------

#     try:
#         # Load full dataset
#         dataset = load_dataset("galileo-ai/ragbench","hotpotqa")  # remove split="train"

#         # Concatenate splits if there are multiple
#         if isinstance(dataset, dict):
#             dataset = concatenate_datasets(list(dataset.values()))  # single Dataset object
#         print(f"[DEBUG] Loaded Hugging Face dataset with {len(dataset)} rows")
#         # Select only needed columns
#         dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["question", "answer"]])

#         # Create a small train/test split 
#         split = dataset.train_test_split(test_size=0.2, shuffle=True)
#         train_dataset = split["train"]
#         test_dataset = split["test"]

#         # Concatenate instruction + response
#         def concat_columns(example):
#             return {
#                 "page_content": f"Question: {example['question']}\nResponse: {example['answer']}"
#             }

#         train_dataset = train_dataset.map(concat_columns)
#         test_dataset = test_dataset.map(concat_columns)

#         # Convert to LangChain Documents
#         docs = [
#             Document(page_content=doc["page_content"])
#             for doc in train_dataset
#         ]

#         print(f"[DEBUG] Loaded {len(docs)} Hugging Face documents")
#         documents.extend(docs)
        
#     except Exception as e:
#         print(f"[ERROR] Failed to load Hugging Face dataset: {e}")




    # ------------------------
    # 2. Load local files
    # ------------------------


#     # PDF files
#     pdf_files = list(data_path.glob('**/*.pdf'))
#     print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
#     for pdf_file in pdf_files:
#         print(f"[DEBUG] Loading PDF: {pdf_file}")
#         try:
#             loader = PyPDFLoader(str(pdf_file))
#             loaded = loader.load()
#             print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
#             documents.extend(loaded)
#         except Exception as e:
#             print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")


#     # TXT files
#     txt_files = list(data_path.glob('**/*.txt'))
#     print(f"[DEBUG] Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")
#     for txt_file in txt_files:
#         print(f"[DEBUG] Loading TXT: {txt_file}")
#         try:
#             loader = TextLoader(str(txt_file))
#             loaded = loader.load()
#             print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
#             documents.extend(loaded)
#         except Exception as e:
#             print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

#     # CSV files
#     csv_files = list(data_path.glob('**/*.csv'))
#     print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
#     for csv_file in csv_files:
#         print(f"[DEBUG] Loading CSV: {csv_file}")
#         try:
#             loader = CSVLoader(str(csv_file))
#             loaded = loader.load()
#             print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
#             documents.extend(loaded)
#         except Exception as e:
#             print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

#     # Excel files
#     xlsx_files = list(data_path.glob('**/*.xlsx'))
#     print(f"[DEBUG] Found {len(xlsx_files)} Excel files: {[str(f) for f in xlsx_files]}")
#     for xlsx_file in xlsx_files:
#         print(f"[DEBUG] Loading Excel: {xlsx_file}")
#         try:
#             loader = UnstructuredExcelLoader(str(xlsx_file))
#             loaded = loader.load()
#             print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {xlsx_file}")
#             documents.extend(loaded)
#         except Exception as e:
#             print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

#     # Word files
#     docx_files = list(data_path.glob('**/*.docx'))
#     print(f"[DEBUG] Found {len(docx_files)} Word files: {[str(f) for f in docx_files]}")
#     for docx_file in docx_files:
#         print(f"[DEBUG] Loading Word: {docx_file}")
#         try:
#             loader = Docx2txtLoader(str(docx_file))
#             loaded = loader.load()
#             print(f"[DEBUG] Loaded {len(loaded)} Word docs from {docx_file}")
#             documents.extend(loaded)
#         except Exception as e:
#             print(f"[ERROR] Failed to load Word {docx_file}: {e}")

#     # JSON files
#     json_files = list(data_path.glob('**/*.json'))
#     print(f"[DEBUG] Found {len(json_files)} JSON files: {[str(f) for f in json_files]}")
#     for json_file in json_files:
#         print(f"[DEBUG] Loading JSON: {json_file}")
#         try:
#             loader = JSONLoader(str(json_file))
#             loaded = loader.load()
#             print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
#             documents.extend(loaded)
#         except Exception as e:
#             print(f"[ERROR] Failed to load JSON {json_file}: {e}")

#     print(f"[DEBUG] Total loaded documents: {len(documents)}")
    
#     return documents

# # Example usage
# if __name__ == "__main__":
#     docs = load_all_documents("data")
#     print(f"Loaded {len(docs)} documents.")
#     print("Example document:", docs[0] if docs else None)




