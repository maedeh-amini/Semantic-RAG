
import os
import json
import requests
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_core.messages import HumanMessage  # only for interface with messages
load_dotenv()


# -----------------------------
# 1 Custom GPT-OSS wrapper
# -----------------------------
class LLMPipeline:
    """
    Minimal wrapper for a self-hosted GPT-OSS model endpoint.
    Compatible with LangChain-like usage with HumanMessage.
    """
    def __init__(self, model_name: str, api_key: str, api_base: str, temperature: float = 0.0):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature

    def generate(self, messages):
        """
        Accepts a list of HumanMessage objects.
        Returns an object with attribute generations[0][0].text
        """
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": m.content} for m in messages],
            "temperature": self.temperature
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(self.api_base, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        # Chat-style response parsing (adjusted)
        try:
            assistant_text = data["choices"][0]["message"]["content"]
        except KeyError:
            # fallback for completion-style endpoint
            assistant_text = data["choices"][0]["text"]

        # Adapt to same interface as LangChain LLM
        class Gen:
            def __init__(self, text):
                self.text = text

        class Res:
            def __init__(self, text):
                self.generations = [[Gen(text)]]

        return Res(assistant_text)
# -----------------------------
# 2 RAG Search class
# -----------------------------
class RAGSearch:
    def __init__(self,
                 persist_dir: str = "faiss_store",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "gpt-oss", #gpt-oss-120b # gpt-oss # gemma3
                 api_key_env: str = "UOS_API_KEY",
                 api_base_env: str = "UOS_API_BASE"):

        # Load pre-built FAISS vector store
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.vectorstore.load()
        print(f"[INFO] FAISS vector store loaded from '{persist_dir}'")

        # Load GPT-OSS API key and base URL
        self.api_key = os.getenv(api_key_env)
        self.api_base = os.getenv(api_base_env)
        if not self.api_key or not self.api_base:
            raise ValueError(f"API key or base URL not found in environment variables '{api_key_env}', '{api_base_env}'")

        # Initialize custom LLM pipeline
        self.llm = LLMPipeline(
            model_name=llm_model,
            api_key=self.api_key,
            api_base=self.api_base
        )
        print(f"[INFO] GPT-OSS LLM initialized: {llm_model} at {self.api_base}")

        # Load external prompt template from project root /prompts/rag_prompt.txt
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # main project root
        prompt_path = os.path.join(root_dir, "prompts", "rag_prompt.txt")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt template file not found: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        print(f"[INFO] Loaded prompt template from {prompt_path}")

    # -----------------------------
    # Search and summarize
    # -----------------------------
    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."

        prompt = self.prompt_template.format(query=query, context=context)
        response = self.llm.generate([HumanMessage(content=prompt)])
        return response.generations[0][0].text

    # -----------------------------
    # DeepEval-compatible query
    # -----------------------------
    def query_rag(self, question: str, top_k: int = 3) -> str:
        results = self.vectorstore.query(question, top_k=top_k)
        retrieved_docs = []
        context_chunks = []

        for r in results:
            metadata = r.get("metadata", {})
            text = metadata.get("text", "")
            if text:
                retrieved_docs.append({"page_content": text})
                context_chunks.append(text)

        context = "\n\n".join(context_chunks)

        if not context:
            answer = "No relevant documents found."
        else:
            prompt = self.prompt_template.format(query=question, context=context)
            response = self.llm.generate([HumanMessage(content=prompt)])
            answer = response.generations[0][0].text

        return json.dumps({
            "answer": answer,
            "retrieved_docs": retrieved_docs
        })


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    rag_search = RAGSearch()

    query = "Dandong and Hotan are located in what country?"

    # Normal usage
    summary = rag_search.search_and_summarize(query, top_k=5)
    print("Summary:", summary)

    # RAGAS-compatible usage
    ragas_output = rag_search.query_rag(query, top_k=5)
    print("DeepEval output:", ragas_output)
