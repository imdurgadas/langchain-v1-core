# create a file: 28_planner_executor.py
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

# -------------------------------------------------------
# Step 1: Define the plan structure (structured output from Part 7)
# -------------------------------------------------------
class ResearchTask(BaseModel):
    """A single step in the research plan."""
    task_id: str = Field(description="Unique ID like 'task_001'")
    title: str = Field(description="Short description of what this step does")
    instructions: str = Field(description="Specific instructions for the executor")
    depends_on: List[str] = Field(
        default=[],
        description="List of task_ids that must complete before this step"
    )
    estimated_minutes: int = Field(description="Estimated time in minutes")

class ResearchPlan(BaseModel):
    """A structured plan for a multi-step investigative research project."""
    story_title: str
    thesis: str = Field(description="The core angle/thesis of the investigation")
    tasks: List[ResearchTask]
    total_estimated_hours: float

# -------------------------------------------------------
# Step 2: The Planner — produces the structured plan
# -------------------------------------------------------
planner_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are the investigative desk editor at Tech News Daily.
Your job is to create research plans for investigative stories.
Each plan must:
- Break the investigation into 4-6 distinct, sequential tasks
- Each task should be completable independently by a researcher
- Include clear instructions for each step
- Estimate realistic time per step"""),
        ("human", """Create a research plan for this investigative story:

Story brief: {story_brief}

The plan will be executed over multiple sessions by an AI research team.
Each task's output will be saved to a file for use by subsequent tasks.""")
    ])
    | llm.with_structured_output(ResearchPlan)
)

# -------------------------------------------------------
# Step 3: Generate a plan for a real Tech News Daily story
# -------------------------------------------------------
story_brief = """
Investigation: "The Hidden Cost of AI Infrastructure — How Hyperscaler GPU Buildouts Are 
Straining Power Grids Across Three Continents"

We want to investigate whether the rapid buildout of AI data centres by AWS, Google, 
Microsoft, and Meta is causing measurable strain on local power infrastructure, 
water usage, and real estate markets in key regions (Virginia, Iowa, Dublin, Singapore).
"""

print("Generating research plan...\n")
plan: ResearchPlan = planner_chain.invoke({"story_brief": story_brief})

print(f"Story: {plan.story_title}")
print(f"Thesis: {plan.thesis}")
print(f"Estimated total: {plan.total_estimated_hours}h")
print(f"\nTasks ({len(plan.tasks)}):")
for task in plan.tasks:
    deps = f" [after: {', '.join(task.depends_on)}]" if task.depends_on else ""
    print(f"  {task.task_id}: {task.title} (~{task.estimated_minutes}min){deps}")

# -------------------------------------------------------
# Step 4: Save the plan to a file — this is the handoff to the executor
# -------------------------------------------------------
os.makedirs("./newsroom_workspace", exist_ok=True)
plan_path = "./newsroom_workspace/investigation_plan.json"

with open(plan_path, "w") as f:
    json.dump(plan.model_dump(), f, indent=2)

print(f"\n✓ Plan saved to: {plan_path}")
print("  The executor will read this file and process tasks in order.")
