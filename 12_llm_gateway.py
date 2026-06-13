# create a file: 12_llm_gateway.py
import time
import os
from dotenv import load_dotenv
from litellm import completion, completion_cost

load_dotenv()

# Priority order: most capable first, cheapest/fastest last
# The gateway tries each in order until one succeeds
FAILOVER_MODELS = [
    "gemini/gemini-3.5-flash",         # Primary: latest stable
    "gemini/gemini-2.5-flash",         # Fallback 1: previous generation, still reliable
    "gemini/gemini-flash-latest",      # Fallback 2: last resort, fastest/cheapest
]

def call_with_failover(user_prompt: str, timeout_seconds: int = 8) -> dict:
    """
    Calls LLM models in priority order, falling back on timeout or error.
    Returns a dict with response content, model used, latency, and cost.
    """
    start = time.time()
    last_error = None

    for model in FAILOVER_MODELS:
        try:
            print(f"Trying model: {model}")
            response = completion(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=timeout_seconds
            )

            latency = time.time() - start
            cost = completion_cost(completion_response=response)

            return {
                "content": response.choices[0].message.content,
                "model_used": model,
                "latency_seconds": round(latency, 3),
                "cost_usd": round(cost, 6),
            }

        except Exception as e:
            last_error = str(e)
            print(f"  Model {model} failed: {e}. Trying next...")
            continue

    raise RuntimeError(
        f"All models in failover chain failed. Last error: {last_error}"
    )


# Run it
result = call_with_failover("Summarise the main trade-offs between SQL and NoSQL databases in 3 bullet points.")

print(f"\n=== Result ===")
print(f"Model used:   {result['model_used']}")
print(f"Latency:      {result['latency_seconds']}s")
print(f"Cost:         ${result['cost_usd']}")
print(f"\nResponse:\n{result['content']}")
