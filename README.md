# LangChain v1.x Core & LangGraph Series - Companion Code

This repository contains the complete, runnable companion code files for the **LangChain v1.x Core & LangGraph** blog series — a comprehensive 10-part guide to building stateful agentic workflows, RAG systems, MCP integrations, prompt engineering, streaming, and evaluations in LangChain.

---

## Repository Structure

To maintain compatibility with both python import rules and the blog post text, the code is structured as follows:
- **Numbered files** (e.g., `01_model_basics.py`): Keep original filenames as referenced in the blog posts for easy matching.
- **Clean-named copies** (e.g., `model_basics.py`): Provide clean python modules to resolve python's syntax limitations (Python throws a `SyntaxError` when importing from module names starting with a digit).

```text
langchain-v1-core/
├── requirements.txt            # Project dependencies
├── README.md                   # Setup and usage guide
├── 01_model_basics.py          # Call model using LangChain API
├── 02_messages.py              # Message handling and chat structures
├── 03_streaming.py             # Real-time token streaming
├── 04_agent.py                 # Basic single-agent with tools
├── 05_langgraph_basic.py       # State-driven LangGraph basics
├── 06_langgraph_memory.py      # Graph with persistent checkpointers
├── 07_rag_ingest.py            # ChromaDB document ingestion pipeline
├── 08_rag_query.py             # Standard vector search retrieval query
├── 09_vectorless_rag.py        # Semantic search with alternative strategies
├── 10_mcp_orchestrator.py      # Model Context Protocol client
├── 11_guardrails.py            # Safety and input filtering gates
├── 12_llm_gateway.py           # Fallback and gateway routing
├── 13_evaluation.py            # Automated LLM-as-a-judge evaluators
├── 14_multi_agent_state.py     # Specialist agents state definition
├── 15_specialist_agents.py     # Individual expert node tasks
├── 16_supervisor.py            # Supervisory routing state machine
├── 17_multi_agent_graph.py     # Complete supervisor workflow assembly
├── 18_multi_agent_memory.py    # Graph with cross-thread shared memory
├── 19_guarded_multi_agent.py   # Multi-agent system with guardrails
├── 20_prompt_templates.py      # Dynamic ChatPromptTemplate usage
├── 21_structured_output.py     # Type-safe structured output extraction
├── 22_output_parsers.py        # Custom response output parsing
├── 23_message_trimming.py      # Token-budget-based message trimming
├── 24_summarisation_fallback.py # Fallback to LLM message summary
├── 25_context_layers.py        # Dynamic multi-tiered context strategy
├── 26_event_streaming.py       # Stream events with astream_events()
├── 27_live_dashboard.py        # Terminal-based real-time event UI
├── 28_planner_executor.py      # LangGraph Planner-Executor flow
├── 29_context_files.py         # Local file ingestion for context
├── 30_resumable_tasks.py       # State checkpointing and resumability
├── mcp_server_data.py          # Sample data provider MCP server
├── mcp_server_network.py       # Sample network provider MCP server
└── [clean_copies].py           # Clean module names (e.g. agent.py)
```

---

## Prerequisites & Installation

1. **Navigate** to the project directory:
   ```bash
   cd langchain-v1-core
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Environment Configuration

Create a `.env` file in the root of this directory:

```env
# Google AI Studio API Key (for Gemini models)
GOOGLE_API_KEY=your_gemini_api_key_here

# LangSmith Tracing (Optional - for Part 5/Part 10 tracing)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=DevPulse-Production
```

---

## How to Run the Scripts

Always ensure your virtual environment is active and environment variables are loaded. You can run any script directly using `python`:

### Example: Running LangGraph Stateful Memory
```bash
python langgraph_memory.py
```

*Note: You can run both the numbered files (e.g., `python 06_langgraph_memory.py`) or the clean-named files (e.g., `python langgraph_memory.py`). Both versions work identically and use the updated import structure.*
