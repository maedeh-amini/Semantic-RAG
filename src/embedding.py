
import sys
print(sys.path)

from typing import List, Any
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from src.data_loader import load_all_documents


class EmbeddingPipeline:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", overlap_ratio: float = 0.2):
        self.overlap_ratio = overlap_ratio
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def sentence_split(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]

    def create_dynamic_chunks(self, documents: List[Any]) -> List[str]:
        """
        Each sentence becomes a chunk.
        Overlap = 20% of previous sentence tokens.
        """
        chunks = []
        prev_tokens = []

        for doc in documents:

            sentences = self.sentence_split(doc.page_content)

            for sent in sentences:
                tokens = sent.split()

                # Compute overlap from previous sentence
                overlap_size = int(len(prev_tokens) * self.overlap_ratio)
                overlap_tokens = prev_tokens[-overlap_size:] if overlap_size > 0 else []

                chunk_tokens = overlap_tokens + tokens
                chunk_text = " ".join(chunk_tokens)

                chunks.append(chunk_text)

                prev_tokens = tokens

        print(f"[INFO] Created {len(chunks)} chunks using sentence-based dynamic overlap.")
        return chunks

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        print(f"[INFO] Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings


# Example usage
if __name__ == "__main__":

    docs = load_all_documents("data")

    emb_pipe = EmbeddingPipeline(overlap_ratio=0.2)

    chunks = emb_pipe.create_dynamic_chunks(docs)

    embeddings = emb_pipe.embed_chunks(chunks)

    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)






#------------------------
# Previous Version
# ------------------------

# import sys
# print(sys.path)



# from typing import List, Any
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from src.data_loader import load_all_documents

# class EmbeddingPipeline:

#     def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.model = SentenceTransformer(model_name)
#         print(f"[INFO] Loaded embedding model: {model_name}")
    
#     def chunk_documents(self, documents: List[Any]) -> List[Any]:
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap,
#             length_function=len,
#             separators=["\n\n", "\n", " ", ""]
#         )
#         chunks = splitter.split_documents(documents)
#         print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
#         return chunks
    
#     def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
#         texts = [chunk.page_content for chunk in chunks]
#         print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
#         embeddings = self.model.encode(texts, show_progress_bar=True)
#         print(f"[INFO] Embeddings shape: {embeddings.shape}")
#         return embeddings
    
#     # Example usage
# if __name__ == "__main__":
    
#     docs = load_all_documents("data")
#     emb_pipe = EmbeddingPipeline()
#     chunks = emb_pipe.chunk_documents(docs)
#     embeddings = emb_pipe.embed_chunks(chunks)
#     print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)