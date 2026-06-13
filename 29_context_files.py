# create a file: 29_context_files.py
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

WORKSPACE = "./newsroom_workspace"
os.makedirs(WORKSPACE, exist_ok=True)

# -------------------------------------------------------
# File-based task executor
# Each task: reads dependencies from files → runs LLM → writes output to file
# -------------------------------------------------------
def get_task_output_path(task_id: str) -> str:
    return os.path.join(WORKSPACE, f"{task_id}_output.md")

def task_is_complete(task_id: str) -> bool:
    """Check if a task has already been completed (output file exists)."""
    return os.path.exists(get_task_output_path(task_id))

def load_dependency_context(depends_on: list[str]) -> str:
    """Load the output of completed dependency tasks as context."""
    context_parts = []
    for dep_id in depends_on:
        path = get_task_output_path(dep_id)
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            context_parts.append(f"=== Output from {dep_id} ===\n{content}")
        else:
            context_parts.append(f"=== {dep_id}: NOT YET COMPLETE ===")
    return "\n\n".join(context_parts) if context_parts else "No prior context."

def execute_task(task: dict) -> str:
    """
    Execute a single research task:
    1. Load dependency context from files
    2. Run the LLM with task instructions + dependency context
    3. Save output to a file
    4. Return the output
    """
    task_id = task["task_id"]
    
    # Skip if already done (resumability)
    if task_is_complete(task_id):
        output_path = get_task_output_path(task_id)
        with open(output_path) as f:
            existing = f.read()
        print(f"  ⏭  {task_id} already complete — skipping")
        return existing
    
    # Load context from dependencies (key: only load what's needed)
    dep_context = load_dependency_context(task.get("depends_on", []))
    
    # Build the prompt with only the relevant context
    # This is context isolation in action: each task sees only its dependencies,
    # not the entire conversation history
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior investigative researcher at Tech News Daily.
Execute the assigned research task thoroughly and factually.
If you cannot verify a specific claim, mark it [NEEDS VERIFICATION].
Structure your output clearly with headings and bullet points."""),
        ("human", """Research Task: {task_title}

Specific instructions:
{instructions}

Context from prior research steps:
{dependency_context}

Produce a detailed research output. This will be saved to a file and used by subsequent tasks.""")
    ])
    
    chain = research_prompt | llm
    
    print(f"  ▶ Executing {task_id}: {task['title']}")
    response = chain.invoke({
        "task_title": task["title"],
        "instructions": task["instructions"],
        "dependency_context": dep_context
    })
    
    output = response.content
    
    # Save to file with metadata header
    output_path = get_task_output_path(task_id)
    with open(output_path, "w") as f:
        f.write(f"# {task['title']}\n")
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Completed: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"---\n\n")
        f.write(output)
    
    print(f"  ✓ {task_id} complete → saved to {output_path}")
    return output


# -------------------------------------------------------
# Load the plan and execute the first two tasks
# (demonstrating partial execution + resumability)
# -------------------------------------------------------
plan_path = os.path.join(WORKSPACE, "investigation_plan.json")

if not os.path.exists(plan_path):
    print("Run 28_planner_executor.py first to generate the plan.")
    exit(1)

with open(plan_path) as f:
    plan = json.load(f)

print(f"\nInvestigation: {plan['story_title']}")
print(f"Tasks to execute: {len(plan['tasks'])}\n")

# Execute tasks in order, respecting dependencies
# (For parallel tasks, you would check depends_on and use asyncio.gather)
for task in plan["tasks"][:2]:  # execute first 2 tasks as a demo
    print(f"\nTask: {task['task_id']} — {task['title']}")
    execute_task(task)

print(f"\n\nWorkspace contents:")
for f in os.listdir(WORKSPACE):
    size = os.path.getsize(os.path.join(WORKSPACE, f))
    print(f"  {f} ({size} bytes)")
