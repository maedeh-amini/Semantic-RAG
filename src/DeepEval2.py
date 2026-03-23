
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
# These are mostly your original settings, unchanged
os.environ["DEEPEVAL_DISABLE_CONFIDENT"] = "false"
os.environ["DEEPEVAL_ASYNC_MODE"] = "true"
os.environ["DEEPEVAL_MAX_CONCURRENCY"] = "2"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1000"  # per-example timeout


# -----------------------------
# Folder for saving intermediate & final results
# -----------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# File where partial test cases are saved incrementally
PARTIAL_RESULTS_FILE = os.path.join(RESULTS_DIR, "partial_results.json")


# --------------------------------------------------
# 1 Load HotpotQA RAGBench dataset
# --------------------------------------------------
def load_test_dataset(limit=None):
    """
    Load HotpotQA dataset from ragbench.
    Only keep the 'question' and 'response' columns.
    Optionally limit the number of examples (for small tests).
    """
    dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["question", "response"]]
    )

    data = [
        {"question": row["question"], "reference": row["response"]}
        for row in dataset
    ]

    if limit:
        data = data[:limit]

    return data


# --------------------------------------------------
# 2 Build DeepEval test cases with checkpointing
# --------------------------------------------------
def build_test_cases(rag_search: RAGSearch, dataset):
    """
    Build DeepEval test cases by querying the RAG system.
    
    NEW FEATURES:
    1. Checkpointing:
       - Loads partial results from 'partial_results.json'
       - Resumes from the last saved example
    2. Skipping failed examples:
       - If query fails, logs a warning and continues
    3. Incremental saving:
       - After each example, the intermediate results are saved to disk
    """

    # Load any partial results to resume from previous run
    if os.path.exists(PARTIAL_RESULTS_FILE):
        with open(PARTIAL_RESULTS_FILE, "r", encoding="utf-8") as f:
            test_cases = [LLMTestCase(**tc) for tc in json.load(f)]
        start_idx = len(test_cases)
        print(f"[INFO] Resuming from example #{start_idx + 1}")
    else:
        test_cases = []
        start_idx = 0

    # Iterate through dataset starting from last saved index
    for i, ex in enumerate(tqdm(dataset[start_idx:], desc="Building DeepEval test cases"), start=start_idx):
        try:
            # Query RAG system for answer and retrieved docs
            rag_output = json.loads(rag_search.query_rag(ex["question"], top_k=5))
            retrieved_docs = rag_output.get("retrieved_docs", [])
            retrieved_texts = [
                doc.get("page_content", "")
                for doc in retrieved_docs
                if doc.get("page_content")
            ]
            answer = rag_output.get("answer", "")

            # Build DeepEval test case object
            test_case = LLMTestCase(
                input=ex["question"],
                expected_output=ex["reference"],
                actual_output=answer,
                retrieval_context=retrieved_texts
            )

            test_cases.append(test_case)

            # -------------------------
            # Incremental saving
            # -------------------------
            # Saves all test cases after every example so progress is not lost
            with open(PARTIAL_RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump([tc.__dict__ for tc in test_cases], f, indent=2, ensure_ascii=False)

        except Exception as e:
            # Skip examples that fail and continue
            print(f"[WARNING] Example #{i + 1} failed: {e}. Skipping...")
            continue

    return test_cases


# --------------------------------------------------
# 3 GPT-OSS Judge Model for DeepEval
# --------------------------------------------------
class GPTOSSJudge(DeepEvalBaseLLM):
    """
    Judge model wrapper for DeepEval.
    Uses your existing LLMPipeline to query gpt-oss or other LLMs.
    """

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
        response = self.pipeline.generate([HumanMessage(content=prompt)])
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
    test_data = load_test_dataset(limit=5)  # set a limit ---> load_test_dataset(limit=1)
    print(f"[INFO] Loaded {len(test_data)} questions")

    print("\n[INFO] Generating RAG answers (with checkpointing)...")
    test_cases = build_test_cases(rag_search, test_data)
    print(f"[INFO] Built {len(test_cases)} test cases")

    print("\n[INFO] Initializing GPT-OSS judge model...")
    judge_model = GPTOSSJudge(model_name="gpt-oss")

    metrics = [
        # AnswerRelevancyMetric(model=judge_model, threshold=0.7),
        # FaithfulnessMetric(model=judge_model, threshold=0.7),
        # ContextualPrecisionMetric(model=judge_model, threshold=0.7),
        # ContextualRecallMetric(model=judge_model, threshold=0.7),
          ContextualRelevancyMetric(model=judge_model, threshold=0.7),
    ]

    print("\n[INFO] Running DeepEval evaluation...\n")
    deepeval.evaluate(test_cases=test_cases, metrics=metrics)
    print("\n[INFO] DeepEval evaluation completed.")

    # -----------------------------
    # Save final results
    # -----------------------------
    final_results_file = os.path.join(RESULTS_DIR, "final_results.json")
    with open(final_results_file, "w", encoding="utf-8") as f:
        json.dump([tc.__dict__ for tc in test_cases], f, indent=2, ensure_ascii=False)
    print(f"[INFO] Final results saved to {final_results_file}")