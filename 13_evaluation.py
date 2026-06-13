# create a file: 13_evaluation.py
import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# -------------------------------------------------------
# Step 1: Create an evaluation dataset in LangSmith
# -------------------------------------------------------
langsmith = Client()
DATASET_NAME = "LangChain Series Eval — Core Concepts"

# Only create the dataset if it does not already exist
if not langsmith.has_dataset(dataset_name=DATASET_NAME):
    dataset = langsmith.create_dataset(
        dataset_name=DATASET_NAME,
        description="Ground truth QA pairs for the LangChain v1.x series evaluation"
    )

    # Each example: an input the agent receives, and the expected correct output
    test_cases = [
        {
            "input": "What is a LangGraph checkpointer used for?",
            "expected": "A checkpointer saves the graph state after every node execution, enabling multi-turn memory and crash recovery."
        },
        {
            "input": "What is the difference between an MCP server with stdio transport versus http transport?",
            "expected": "stdio runs the server as a local subprocess communicating via stdin/stdout. http runs it as a web service accessible over a network."
        },
        {
            "input": "Why should you use add_messages as a reducer in LangGraph state?",
            "expected": "add_messages appends new messages to the existing list rather than replacing it, preserving the full conversation history across graph cycles."
        },
    ]

    for case in test_cases:
        langsmith.create_example(
            inputs={"question": case["input"]},
            outputs={"expected_answer": case["expected"]},
            dataset_id=dataset.id
        )
    print(f"Created dataset '{DATASET_NAME}' with {len(test_cases)} examples")
else:
    print(f"Dataset '{DATASET_NAME}' already exists — skipping creation")


# -------------------------------------------------------
# Step 2: Define the application being evaluated
# -------------------------------------------------------
def agent_under_test(inputs: dict) -> dict:
    """
    This is the function we are evaluating.
    In production, this would be your actual agent pipeline.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    response = llm.invoke(inputs["question"])
    return {"answer": response.content}


# -------------------------------------------------------
# Step 3: Define the evaluator (LLM as a Judge)
# -------------------------------------------------------
def evaluate_correctness(run_outputs: dict, example_outputs: dict) -> dict:
    """
    An LLM grades whether the agent's answer is factually consistent
    with the expected answer.
    
    Using a capable model (gemini-3.5-flash) as the judge ensures nuanced
    scoring — partial credit for mostly-correct answers is possible
    by returning a score between 0 and 1.
    """
    judge = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

    prompt = f"""You are a strict technical evaluator.

Expected answer: {example_outputs['expected_answer']}
Agent's answer: {run_outputs['answer']}

Does the agent's answer correctly capture the key technical facts in the expected answer?
Respond with ONLY a number: 1 (correct/mostly correct) or 0 (wrong/missing key facts)."""

    verdict = judge.invoke(prompt).content.strip()

    try:
        score = int(verdict[0])  # take first character in case of trailing whitespace
    except (ValueError, IndexError):
        score = 0  # default to fail on malformed judge response

    return {"key": "factual_correctness", "score": score}


# -------------------------------------------------------
# Step 4: Run the evaluation
# -------------------------------------------------------
from langsmith import evaluate

print("\nRunning evaluation...")
results = evaluate(
    agent_under_test,
    data=DATASET_NAME,
    evaluators=[evaluate_correctness],
    experiment_prefix="gemini-flash-baseline",
)

# Print a summary
print("\n=== Evaluation Complete ===")
print("View full results at: https://smith.langchain.com")
