# create a file: 30_resumable_tasks.py
import os
import json
import asyncio
from dotenv import load_dotenv
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

WORKSPACE = "./newsroom_workspace"

# -------------------------------------------------------
# State for the long-horizon investigation graph
# -------------------------------------------------------
class InvestigationState(TypedDict):
    story_brief: str
    plan: dict              # the full plan dict
    completed_tasks: List[str]  # task_ids that are done
    current_task_id: str    # which task is running
    final_report: str       # assembled at the end
    status: str             # "planning" | "executing" | "complete"

# -------------------------------------------------------
# Node: Load or generate the plan
# -------------------------------------------------------
def planning_node(state: InvestigationState) -> dict:
    plan_path = os.path.join(WORKSPACE, "investigation_plan.json")
    
    if os.path.exists(plan_path):
        print("\n[Planning] Existing plan found — loading")
        with open(plan_path) as f:
            plan = json.load(f)
    else:
        print("\n[Planning] Generating new plan...")
        # In production: call the planner_chain from Part 1
        # Here we create a minimal plan for demonstration
        plan = {
            "story_title": "AI Infrastructure Power Grid Impact",
            "thesis": "Hyperscaler AI buildouts are straining power infrastructure",
            "tasks": [
                {
                    "task_id": "task_001",
                    "title": "Power consumption baseline",
                    "instructions": "Research current power consumption of major AI data centres. Focus on AWS us-east-1, Google Council Bluffs Iowa, and Microsoft Dublin.",
                    "depends_on": [],
                    "estimated_minutes": 30
                },
                {
                    "task_id": "task_002", 
                    "title": "Grid impact analysis",
                    "instructions": "Based on power consumption data, analyse reported grid strain incidents. Look for utility company statements, regulatory filings, and news reports from 2024-2026.",
                    "depends_on": ["task_001"],
                    "estimated_minutes": 45
                },
                {
                    "task_id": "task_003",
                    "title": "Executive summary",
                    "instructions": "Synthesise all research into a 500-word executive summary suitable for a Tech News Daily front-page investigation.",
                    "depends_on": ["task_001", "task_002"],
                    "estimated_minutes": 20
                }
            ],
            "total_estimated_hours": 1.58
        }
        os.makedirs(WORKSPACE, exist_ok=True)
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2)
    
    return {"plan": plan, "status": "executing", "completed_tasks": []}

# -------------------------------------------------------
# Node: Execute the next pending task
# -------------------------------------------------------
def executor_node(state: InvestigationState) -> dict:
    plan = state["plan"]
    completed = set(state.get("completed_tasks", []))
    
    # Find the next task whose dependencies are all satisfied
    next_task = None
    for task in plan["tasks"]:
        if task["task_id"] in completed:
            continue
        deps_met = all(d in completed for d in task.get("depends_on", []))
        if deps_met:
            next_task = task
            break
    
    if next_task is None:
        print("\n[Executor] All tasks complete")
        return {"status": "assembling"}
    
    task_id = next_task["task_id"]
    task_output_path = os.path.join(WORKSPACE, f"{task_id}_output.md")
    
    # Resumability: skip if already done
    if os.path.exists(task_output_path):
        print(f"\n[Executor] {task_id} already complete — skipping")
        new_completed = list(completed) + [task_id]
        return {
            "completed_tasks": new_completed,
            "current_task_id": task_id
        }
    
    # Load dependency context (context isolation — only load what's needed)
    dep_context = ""
    for dep_id in next_task.get("depends_on", []):
        dep_path = os.path.join(WORKSPACE, f"{dep_id}_output.md")
        if os.path.exists(dep_path):
            with open(dep_path) as f:
                dep_context += f"\n\n=== {dep_id} findings ===\n{f.read()}"
    
    # Execute
    print(f"\n[Executor] Running {task_id}: {next_task['title']}")
    research_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are an investigative researcher at Tech News Daily. Be thorough and factual. Mark unverified claims as [NEEDS VERIFICATION]."),
            ("human", "Task: {title}\n\nInstructions: {instructions}\n\nPrior context:\n{context}")
        ])
        | llm
    )
    response = research_chain.invoke({
        "title": next_task["title"],
        "instructions": next_task["instructions"],
        "context": dep_context or "No prior context — this is the first task."
    })
    
    # Save to file
    from datetime import datetime
    with open(task_output_path, "w") as f:
        f.write(f"# {next_task['title']}\nCompleted: {datetime.utcnow().isoformat()}Z\n---\n\n")
        f.write(response.content)
    print(f"[Executor] ✓ {task_id} saved to {task_output_path}")
    
    new_completed = list(completed) + [task_id]
    return {
        "completed_tasks": new_completed,
        "current_task_id": task_id
    }

# -------------------------------------------------------
# Node: Assemble the final report from all task outputs
# -------------------------------------------------------
def assembler_node(state: InvestigationState) -> dict:
    print("\n[Assembler] Compiling final report...")
    plan = state["plan"]
    
    all_findings = ""
    for task in plan["tasks"]:
        path = os.path.join(WORKSPACE, f"{task['task_id']}_output.md")
        if os.path.exists(path):
            with open(path) as f:
                all_findings += f"\n\n{f.read()}"
    
    report_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are the editor-in-chief of Tech News Daily."),
            ("human", f"Story thesis: {plan['thesis']}\n\nResearch findings:\n{all_findings}\n\nWrite a 400-word final investigative article suitable for publication.")
        ])
        | llm
    )
    report = report_chain.invoke({})
    
    report_path = os.path.join(WORKSPACE, "final_report.md")
    with open(report_path, "w") as f:
        f.write(f"# {plan['story_title']}\n\n")
        f.write(report.content)
    
    print(f"[Assembler] ✓ Final report saved to {report_path}")
    return {"final_report": report.content, "status": "complete"}

# -------------------------------------------------------
# Routing functions
# -------------------------------------------------------
def route_after_executor(state: InvestigationState) -> str:
    if state.get("status") == "assembling":
        return "assembler"
    plan = state["plan"]
    completed = set(state.get("completed_tasks", []))
    all_done = all(t["task_id"] in completed for t in plan["tasks"])
    return "assembler" if all_done else "executor"

# -------------------------------------------------------
# Build the graph
# -------------------------------------------------------
investigation = StateGraph(InvestigationState)
investigation.add_node("planner", planning_node)
investigation.add_node("executor", executor_node)
investigation.add_node("assembler", assembler_node)
investigation.add_edge(START, "planner")
investigation.add_edge("planner", "executor")
investigation.add_conditional_edges("executor", route_after_executor,
    {"executor": "executor", "assembler": "assembler"})
investigation.add_edge("assembler", END)

# MemorySaver stores the graph state — in production use SqliteSaver
checkpointer = MemorySaver()
investigation_graph = investigation.compile(checkpointer=checkpointer)

# -------------------------------------------------------
# Session 1: Start the investigation (may be interrupted)
# -------------------------------------------------------
thread_config = {"configurable": {"thread_id": "ai-power-investigation-v1"}}
initial_state = {
    "story_brief": "Investigate the impact of AI data centre buildouts on power grids",
    "plan": {},
    "completed_tasks": [],
    "current_task_id": "",
    "final_report": "",
    "status": "planning"
}

print("=" * 60)
print("Session 1: Starting investigation")
print("=" * 60)

final = investigation_graph.invoke(initial_state, thread_config, {"recursion_limit": 30})

print(f"\n{'=' * 60}")
print("Investigation complete!")
print(f"Status: {final['status']}")
print(f"Tasks completed: {final['completed_tasks']}")
print(f"\nFinal report preview:")
print(final["final_report"][:500] + "...")

# -------------------------------------------------------
# Session 2: Demonstrate resumability
# Run again — all tasks are skipped because their output files exist
# -------------------------------------------------------
print(f"\n{'=' * 60}")
print("Session 2: Resuming (simulating restart)")
print("=" * 60)

# Reset completed_tasks to force re-evaluation — graph will find the files and skip
resume_state = {**initial_state, "status": "planning"}
final2 = investigation_graph.invoke(resume_state, thread_config, {"recursion_limit": 30})
print(f"Resume complete. Tasks executed from scratch: 0 (all skipped via file check)")
