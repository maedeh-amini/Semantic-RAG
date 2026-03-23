# import sys
# import os
# import json
# import pytest
# # import warnings
# # warnings.filterwarnings("ignore", category=DeprecationWarning)
# from ragas.metrics.collections import ContextPrecision
# # from ragas.metrics import LLMContextPrecisionWithReference
# from ragas import SingleTurnSample
# from src.search import RAGSearch
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# # Context Precision
# # Definition: Measures how much of the retrieved context contributes to the correct answer.
# # High Precision = Most of the retrieved contexts are useful for answering the query.
# # Low Precision = Many of the retrieved contexts are irrelevant or only partially relevant.

# @pytest.mark.asyncio
# async def test_context_precision(langchain_llm_ragas_wrapper, get_question, print_log, get_ground_truth):
#     """
#     Test the context precision of RAG retrieval against ground-truth answers.
    
# #     Fixtures:
# #         - langchain_llm_ragas_wrapper: LLM wrapped for RAGAS
# #         - get_question: returns a test question
# #         - print_log: logging utility
# #         - get_ground_truth: returns the correct answer for a question
# #     """

#     # Get Question 
#     question = get_question("questions", "simple")

#     # Get Ground-Truth Answer
#     ground_truth_answer = get_ground_truth(question)

#     # Get Response from RAG pipeline
#     response = RAGSearch(question)
#     parsed_response = json.loads(response)

# #     # Initialize the LLM and Ragas Setup for Context Precision 
#     context_precision = ContextPrecision(llm=langchain_llm_ragas_wrapper)


# #     # Prepare the sample for evaluation
#     sample = SingleTurnSample(
#         user_input=question,
#         response=parsed_response.get("answer", " "), 
#         retrieved_contexts=[
#             doc.get("page_content", "") for doc in parsed_response.get("retrieved_docs", [])
#             ], 
#         reference=ground_truth_answer
#         )

#     # Compute the score asynchronously
#     score = await context_precision.single_turn_ascore(sample)

#     # Print/log details for inspection
#     print_log(
#         question=question,
#         model_response=parsed_response.get("answer"),
#         retrieved_contexts=parsed_response.get("retrieved_docs", []),
#         score=score
#     )


# #     # print_log(question, parsed_response["answer"], parsed_response["retrieved_docs"], score=score)
# #     # assert score >= 0.5

#     # assertion
#     assert score >= 0.5, f"Context precision too low: {score}"






import json
import pytest
from ragas.metrics.collections import ContextPrecision
from src.search import RAGSearch


@pytest.mark.asyncio
async def test_context_precision(
    langchain_llm_ragas_wrapper,
    get_question,
    print_log,
    get_ground_truth,
):
    # 1. Question
    question = get_question("questions", "simple")

    # 2. Ground truth
    ground_truth_answer = get_ground_truth(question)

    # 3. Run RAG
    rag = RAGSearch()
    response = rag.query_rag(question)
    parsed_response = json.loads(response)

    retrieved_contexts = [
        doc["page_content"]
        for doc in parsed_response.get("retrieved_docs", [])
    ]

    # 4. RAGAS metric
    context_precision = ContextPrecision(
        llm=langchain_llm_ragas_wrapper
    )

    # 5. Score
    result = await context_precision.ascore(
        user_input=question,
        reference=ground_truth_answer,
        retrieved_contexts=retrieved_contexts,
    )

    score = result.value

    # 6. Logging
    print_log(
        question=question,
        model_response=parsed_response.get("answer"),
        retrieved_contexts=retrieved_contexts,
        score=score,
    )

    # 7. Assertion
    assert score >= 0.5, f"Context precision too low: {score}"
