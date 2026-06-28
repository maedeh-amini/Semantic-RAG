# Semantic Retrieval-Augmented Generation (RAG) Pipeline

A Python-based framework designed for building, optimizing, and evaluating dense vector-based Semantic RAG pipelines. This repository leverages text embeddings and semantic search engines to retrieve contextually relevant information, mitigating LLM hallucinations and improving response faithfulness.


# 🚀 Features

  -    **Semantic Search & Embedding**:  Implements dense vector retrieval utilizing FAISS for local indexing and Typesense for scalable, typo-tolerant semantic search.

  -    **Context-Aware Generation**: Feeds high-relevance semantic context into LLMs using structured, version-controlled prompt templates.

  -    **Rigorous Evaluation**: Integrated with DeepEval to benchmark key RAG metrics, including faithfulness, answer relevance, and hallucination metrics.

  -    **Modern Tooling**: Uses `uv` for lightning-fast dependency management and strict environment reproducibility.


# 📂 Project Structure

```plaintext
├── .deepeval/             # DeepEval configuration and test logs for RAG assessment
├── .vscode/               # Workspace settings
├── data/
│   └── vector_store/      # Local storage for embedded document chunks
├── faiss_store/           # FAISS index files for semantic vector search
├── notebook/              # Jupyter notebooks for experimentation
│   └── typesense.ipynb    # Prototyping semantic search with Typesense
├── prompts/               # Centralized system and user prompt templates
├── src/                   # Core application source code
│   └── vectorstore.py     # Embeddings generation and vector database management
├── .env                   # Environment variables (API keys, DB endpoints)
├── .gitignore             # Git ignore rules
├── .python-version        # Specified Python runtime environment
├── app.py                 # Application entry point (e.g., UI or API layer)
├── main.py                # Main execution script to run the Semantic RAG pipeline
├── pyproject.toml         # Project metadata and dependencies
├── requirements.txt       # Standard pip dependency list
└── uv.lock                # UV lockfile for deterministic builds
```


# 🛠️ Setup & Installation
**Prerequisites**

-  Python (defined in `.python-version`)

-  uv (recommended) or `pip`



# 1. Clone the Repository
```
git clone https://github.com/maedeh-amini/your-repo-name.git
cd your-repo-name
```

# 2. Environment Configuration

Create a `.env` file in the root directory and add your credentials:

```
# LLM and Embedding Provider Keys
OPENAI_API_KEY=your_openai_api_key

# Semantic Search Database Configurations
ACADEMIC_KEY=your_academic_credentials
TYPESENSE_API_KEY=your_typesense_key
TYPESENSE_HOST=localhost
TYPESENSE_PORT=8108
```


# 3. Install Dependencies

Using **uv** (Recommended):
```
Bash

uv sync
```

Using **pip**:
```
pip install -r requirements.txt
```



# 💻 Usage
## **1. Run the Semantic RAG Pipeline**

To run the end-to-end ingestion, retrieval, and generation pipeline:
```
Bash

python main.py
```

## **2. Launch the Application Interface**

If your `app.py` serves an interactive demo (e.g., Streamlit or Gradio):
```
Bash

python -m streamlit run app.py
```

## **3. Run Evaluation Suites**

To evaluate the retrieval accuracy and response generation quality using DeepEval:
```
Bash

deepeval test run
```

# 🧪 Pipeline Architecture

1. **Ingestion & Chunking:** Documents are parsed, split into optimal semantic chunks, and transformed into dense vector embeddings.

2. **Indexing:** Embeddings are indexed using FAISS or Typesense for fast, high-dimensional similarity matching.

3. **Retrieval:** User queries are embedded on the fly to fetch the top-k most semantically relevant context chunks.

4. **Generation:** The retrieved context and user query are compiled into a prompt template and synthesized by the LLM.
