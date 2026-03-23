
import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset

import deepeval
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

from deepeval.models.base_model import DeepEvalBaseLLM

from src.search import RAGSearch, LLMPipeline
from langchain_core.messages import HumanMessage

load_dotenv()

# -----------------------------
# DeepEval environment settings
# -----------------------------
os.environ["DEEPEVAL_DISABLE_CONFIDENT"] = "false"
os.environ["DEEPEVAL_ASYNC_MODE"] = "true"
os.environ["DEEPEVAL_MAX_CONCURRENCY"] = "3"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "10000"


# --------------------------------------------------
# 1 Load HotpotQA RAGBench dataset
# --------------------------------------------------
def load_test_dataset(limit=None):

    dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")

    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["question", "response"]]
    )

    data = [
        {
            "question": row["question"],
            "reference": row["response"]
        }
        for row in dataset
    ]

    if limit:
        data = data[:limit]

    return data


# --------------------------------------------------
# 2 Build DeepEval test cases
# --------------------------------------------------
def build_test_cases(rag_search: RAGSearch, dataset):

    test_cases = []

    for ex in tqdm(dataset, desc="Building DeepEval test cases"):

        rag_output = json.loads(
            rag_search.query_rag(ex["question"], top_k=5)
        )

        retrieved_docs = rag_output.get("retrieved_docs", [])

        retrieved_texts = [
            doc.get("page_content", "")
            for doc in retrieved_docs
            if doc.get("page_content")
        ]

        answer = rag_output.get("answer", "")

        test_cases.append(
            LLMTestCase(
                input=ex["question"],
                expected_output=ex["reference"],
                actual_output=answer,
                retrieval_context=retrieved_texts
            )
        )

    return test_cases


# --------------------------------------------------
# 3 GPT-OSS Judge Model for DeepEval
# --------------------------------------------------
class GPTOSSJudge(DeepEvalBaseLLM):

    def __init__(self,
                 model_name="gpt-oss",
                 api_key_env="UOS_API_KEY",
                 api_base_env="UOS_API_BASE"):

        self.api_key = os.getenv(api_key_env)
        self.api_base = os.getenv(api_base_env)
        self.model_name = model_name

        if not self.api_key or not self.api_base:
            raise ValueError("Missing API credentials in .env")

        self.pipeline = LLMPipeline(
            model_name=self.model_name,
            api_key=self.api_key,
            api_base=self.api_base
        )

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:

        response = self.pipeline.generate(
            [HumanMessage(content=prompt)]
        )

        return response.generations[0][0].text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


# --------------------------------------------------
# 4 Run DeepEval
# --------------------------------------------------
if __name__ == "__main__":

    print("\n[INFO] Initializing RAG system...")
    rag_search = RAGSearch()

    print("\n[INFO] Loading dataset...")

    # IMPORTANT: small test first
    test_data = load_test_dataset()

    print(f"[INFO] Loaded {len(test_data)} questions")

    print("\n[INFO] Generating RAG answers...")
    test_cases = build_test_cases(rag_search, test_data)

    print(f"[INFO] Built {len(test_cases)} test cases")

    print("\n[INFO] Initializing GPT-OSS judge model...")
    judge_model = GPTOSSJudge(model_name="gpt-oss")

    metrics = [

        AnswerRelevancyMetric(
            model=judge_model,
            threshold=0.7
        ),

        FaithfulnessMetric(
            model=judge_model,
            threshold=0.7
        ),

        ContextualPrecisionMetric(
            model=judge_model,
            threshold=0.7
        ),

        ContextualRecallMetric(
            model=judge_model,
            threshold=0.7
        ),

        ContextualRelevancyMetric(
            model=judge_model,
            threshold=0.7
        ),

    ]

    print("\n[INFO] Running DeepEval evaluation...\n")

    deepeval.evaluate(
        test_cases=test_cases,
        metrics=metrics
    )

    print("\n[INFO] DeepEval evaluation completed.")







# Old Version 3:
# import os
# import json
# from datasets import load_dataset
# from tqdm import tqdm
# import deepeval
# # from deepeval.dataset import EvaluationDataset
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import AnswerRelevancyMetric
# from deepeval.metrics import FaithfulnessMetric
# from deepeval.metrics import ContextualPrecisionMetric
# from deepeval.metrics import ContextualRecallMetric
# from deepeval.metrics import ContextualRelevancyMetric
# from deepeval.metrics import HallucinationMetric
# # from deepeval.models import OllamaModel
# from src.search import RAGSearch
# from src.search import LLMPipeline
# from dotenv import load_dotenv
# load_dotenv()

# # -----------------------------
# # 0 DeepEval environment variables
# # -----------------------------
# os.environ["DEEPEVAL_DISABLE_CONFIDENT"] = "true"
# os.environ["DEEPEVAL_ASYNC_MODE"] = "false"
# os.environ["DEEPEVAL_MAX_CONCURRENCY"] = "1"
# os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "10000"


# # --------------------------------------------------
# # 1. Load HF test dataset
# # --------------------------------------------------
# def load_test_dataset():
#     dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")

#     if isinstance(dataset, dict):
#         dataset = dataset["train"]

#     # Only keep the columns needed
#     dataset = dataset.remove_columns(
#         [col for col in dataset.column_names if col not in ["question", "response"]]
#     )

#     # split = dataset.train_test_split(test_size=test_size, shuffle=True)
#     # test_dataset = split["test"]

#     return [
#         {
#             "question": row["question"],
#             "reference": row["response"]
#         }
#         for row in dataset
#     ]


# # --------------------------------------------------
# # 2. Build DeepEval test cases 
# # --------------------------------------------------
# def build_test_cases(rag_search: RAGSearch, dataset):
#     test_cases = []

#     for ex in tqdm(dataset, desc="Building DeepEval test cases"):
#         # Retrieve RAG output
#         rag_output = json.loads(
#             rag_search.query_rag(ex["question"], top_k=5)
#         )

#         retrieved_docs = rag_output.get("retrieved_docs", [])
#         retrieved_texts = [doc["page_content"] for doc in retrieved_docs]
#         answer=rag_output.get("answer","")

#         test_cases.append(
#             LLMTestCase(
#                 input=ex["question"],
#                 expected_output=ex["reference"],
#                 actual_output=answer,
#                 retrieval_context=retrieved_texts  
#             )
#         )

#     return test_cases

# # -----------------------------
# # 3 LLM wrapper for DeepEval
# # -----------------------------
# class LLM_Judge:
#     """
#     Wrapper for DeepEval to use LLMs as the judge model.
#     Must implement a `generate` method that returns text for evaluation.
#     """
#     def __init__(self, model_name: str = "gpt-oss", api_key_env: str = "UOS_API_KEY", api_base_env: str = "UOS_API_BASE"):
#         self.api_key = os.getenv(api_key_env)
#         self.api_base = os.getenv(api_base_env)
#         self.model_name = model_name

#         if not self.api_key or not self.api_base:
#             raise ValueError(f"API key or base URL not found in environment variables '{api_key_env}', '{api_base_env}'")

#         self.pipeline = LLMPipeline(
#             model_name=self.model_name,
#             api_key=self.api_key,
#             api_base=self.api_base
#         )

#     def __call__(self, prompt: str, **kwargs):
#         """
#         DeepEval expects callable with signature: prompt -> text
#         """
#         from langchain_core.messages import HumanMessage
#         response = self.pipeline.generate([HumanMessage(content=prompt)])
#         return response.generations[0][0].text


# # --------------------------------------------------
# # 4. Run DeepEval
# # --------------------------------------------------

# if __name__ == "__main__":
#     # Load RAG searcher
#     rag_search = RAGSearch()

#     # Load test dataset
#     test_data = load_test_dataset()
#     test_cases = build_test_cases(rag_search, test_data)

#     print(f"[DEBUG] Number of test_cases: {len(test_cases)}")

#     # Initialize judge model
#     judge_model = LLM_Judge(model_name="gpt-4.1")
    
#     # Define metrics
#     metrics = [
#         AnswerRelevancyMetric(model=judge_model, threshold=0.7), #threshold defined as documentation
#         FaithfulnessMetric(model=judge_model, threshold=0.7),
#         ContextualPrecisionMetric(model=judge_model, threshold=0.7),
#         ContextualRecallMetric(model=judge_model, threshold=0.7),
#         ContextualRelevancyMetric(model=judge_model, threshold=0.7),
#         HallucinationMetric(model=judge_model, threshold=0.7)
#     ]
    
#     # Run DeepEval
#     deepeval.evaluate(
#         test_cases=test_cases,
#         metrics=metrics
#     )

#     print("\n[INFO] DeepEval evaluation completed.")





# Old Version 2:

# import os
# import json
# from tqdm import tqdm
# from dotenv import load_dotenv

# # Disable DeepEval confident to avoid unnecessary telemetry/logging
# os.environ["DEEPEVAL_DISABLE_CONFIDENT"] = "true"
# os.environ["DEEPEVAL_ASYNC_MODE"] = "false"
# os.environ["DEEPEVAL_MAX_CONCURRENCY"] = "1"
# os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "10000"

# import deepeval
# from datasets import load_dataset as hf_load_dataset
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import AnswerRelevancyMetric
# from deepeval.models.gpt_model import GPTModel # Standard choice, or use Ollama/Anthropic
# from src.search import RAGSearch

# load_dotenv()

# # --------------------------------------------------
# # 1. Load HF test dataset
# # --------------------------------------------------
# def get_ragbench_data():
#     """
#     Loads the ragbench dataset and renames to avoid conflict 
#     with the imported load_dataset function.
#     """
#     dataset = hf_load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")

#     # If dataset is returned as a DatasetDict, grab the relevant split
#     if hasattr(dataset, "column_names"):
#         cols_to_remove = [col for col in dataset.column_names if col not in ["question", "response"]]
#         dataset = dataset.remove_columns(cols_to_remove)

#     return [
#         {
#             "question": row["question"],
#             "reference": row["response"]
#         }
#         for row in dataset
#     ]

# # --------------------------------------------------
# # 2. Build test cases 
# # --------------------------------------------------
# def build_test_cases(rag_search: RAGSearch, test_dataset):
#     test_cases = []

#     for ex in tqdm(test_dataset, desc="Building DeepEval test cases"):
#         # Assuming query_rag returns a JSON string as per your snippet
#         rag_output_raw = rag_search.query_rag(ex["question"], top_k=5)
#         rag_output = json.loads(rag_output_raw)

#         retrieved_docs = rag_output.get("retrieved_docs", [])
#         retrieved_texts = [doc.get("page_content", "") for doc in retrieved_docs]
#         answer = rag_output.get("answer", "")

#         test_case = LLMTestCase(
#             input=ex["question"],
#             actual_output=answer,
#             expected_output=ex["reference"],
#             retrieval_context=retrieved_texts  
#         )
#         test_cases.append(test_case)

#     return test_cases

# # --------------------------------------------------
# # 3. Execution
# # --------------------------------------------------
# if __name__ == "__main__":
#     # Initialize your RAG system
#     rag_engine = RAGSearch()

#     # Prep data
#     test_data = get_ragbench_data()
#     # Note: You might want to slice test_data[:10] for a quick initial smoke test
#     test_cases = build_test_cases(rag_engine, test_data)

#     print(f"[DEBUG] Number of test_cases: {len(test_cases)}")

#     # Define the Evaluation Model
#     # Since you commented out Ollama, I'll default to GPT-4o-mini (cost-effective for eval)
#     # Ensure OPENAI_API_KEY is in your .env
#     eval_model = GPTModel(model="gpt-4o-mini") 

#     # Define Metrics
#     metrics = [
#         AnswerRelevancyMetric(
#             model=eval_model, 
#             threshold=0.7, 
#             include_reason=True
#         )
#     ]

#     # Run Evaluation
#     deepeval.evaluate(
#         test_cases=test_cases,
#         metrics=metrics,
#         print_results=True
#     )

#     print("\n[INFO] DeepEval evaluation completed.")








#Old Version 1:

# import os
# import sys
# from pathlib import Path
# from typing import List,Dict

# from dotenv import load_dotenv

# from deepeval.dataset import EvaluationDataset,Golden
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import (
#     AnswerRelevancyMetric,
#     FaithfulnessMetric,
#     ContextualPrecisionMetric,
#     ContextualRecallMetric,
#     ContextualRelevancyMetric,
#     HallucinationMetric,
# )
# from deepeval import evaluate

# # from logger import GLOBAL_LOGGER as log

# # Reuse your project components
# from src.document_ingestion.data_ingestion import ChatIngestor
# from src.document_chat.retrieval import ConversationalRAG


# # Configurations (mirrors notebook and runners)
# DEEPEVAL_INPUT_DIR = os.getenv("DEEPEVAL_INPUT_DIR", "data_deep_eval")
# # Defaults align with the notebook usage
# UPLOAD_BASE = os.getenv("UPLOAD_BASE", "notebook/eval_data")
# FAISS_BASE = os.getenv("FAISS_BASE", "notebook/eval_faiss_index")
# FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")
# DATASET_ALIAS = os.getenv("DEEPEVAL_DATASET_ALIAS", "test_doc_chat")


# def query_rag(question: str, session_id: str, k: int = 5) -> dict:
#     index_dir = os.path.join(FAISS_BASE, session_id)
#     if not os.path.isdir(index_dir):
#         raise FileNotFoundError(f"FAISS index not found at: {index_dir}")

#     rag = ConversationalRAG(session_id=session_id)
#     rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)
#     answer = rag.invoke(question, chat_history=[])
#     context = rag.get_retrieved_context(question, k=k)
#     return {"answer": answer, "context": context}

# def ensure():







# from dotenv import load_dotenv
# import os

# # Load environment variables BEFORE importing modules that use them
# # Use override=True to ensure environment variables are updated
# load_dotenv(override=True)

# import json
# from datasets import load_dataset
# from tqdm import tqdm

# import deepeval
# from deepeval.dataset import EvaluationDataset
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import (
#     AnswerRelevancyMetric,
#     # FaithfulnessMetric,
#     ContextualPrecisionMetric
#     # ContextualRecallMetric,
#     # ContextualRelevancyMetric,
#     # HallucinationMetric
# )

# from src.search import RAGSearch

# load_dotenv()


# # --------------------------------------------------
# # 1. Load HF test dataset
# # --------------------------------------------------

# def load_test_dataset(test_size=0.001):
#     dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

#     if isinstance(dataset, dict):
#         dataset = dataset["train"]

#     dataset = dataset.remove_columns(
#         [col for col in dataset.column_names if col not in ["instruction", "response"]]
#     )

#     split = dataset.train_test_split(test_size=test_size, shuffle=True)
#     test_dataset = split["test"]

#     return [
#         {
#             "question": row["instruction"],
#             "reference": row["response"]
#         }
#         for row in test_dataset
#     ]


# # --------------------------------------------------
# # 2. Build test cases (FIXED)
# # --------------------------------------------------

# def build_test_cases(rag_search: RAGSearch, test_examples):
#     test_cases = []

#     for ex in tqdm(test_examples, desc="Building DeepEval test cases"):
#         rag_output = json.loads(
#             rag_search.query_rag(ex["question"], top_k=3)
#         )

#         retrieved_docs = rag_output.get("retrieved_docs", [])
        

#         #REQUIRED for FaithfulnessMetric
#         retrieval_context = [
#             doc.get("page_content", "")
#             for doc in retrieved_docs
#             if doc.get("page_content")
#         ]

#         test_cases.append(
#             LLMTestCase(
#                 input=ex["question"],
#                 actual_output=rag_output.get("answer", ""),
#                 expected_output=ex["reference"],
#                 retrieval_context=retrieval_context  #  FIX
#             )
#         )

#     return test_cases


# # --------------------------------------------------
# # 3. Optional: push dataset (cloud only)
# # --------------------------------------------------

# def push_dataset(test_cases):
#     dataset = EvaluationDataset()
#     for tc in test_cases:
#         dataset.add_test_case(tc)

#     dataset.push("bitext-rag-eval")


# # --------------------------------------------------
# # 4. Run DeepEval
# # --------------------------------------------------

# if __name__ == "__main__":
#     rag_search = RAGSearch()

#     test_data = load_test_dataset()
#     test_cases = build_test_cases(rag_search, test_data)

#     print(f"[DEBUG] Type of test_cases: {type(test_cases)}")
#     print(f"[DEBUG] Number of test_cases: {len(test_cases)}")

#     metrics = [
#         AnswerRelevancyMetric(threshold=0.5),
#         # ContextualRelevancyMetric(threshold=0.5),
#         # FaithfulnessMetric(threshold=0.5),
#         ContextualPrecisionMetric(threshold=0.5),
#         # ContextualRecallMetric(threshold=0.5),
#         # HallucinationMetric(threshold=0.5),
#     ]

#     deepeval.evaluate(
#         test_cases=test_cases,
#         metrics=metrics
#     )

#     print("\n[INFO] DeepEval evaluation completed.")






