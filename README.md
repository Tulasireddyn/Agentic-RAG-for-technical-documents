# Agentic RAG for Technical Papers (cs.AI)

This system is an autonomous research assistant that answers questions about recent AI research papers from arXiv (last 90 days). It is built with **LangGraph**, **Ollama**, and **ChromaDB**.

## Key Features

1.  **Agentic Loop (LangGraph)**: The system uses a state machine to decide between retrieval, clarification, refusal, or answering.
2.  **RAG-Fusion**: Implements multi-query generation and Reciprocal Rank Fusion (RRF) to overcome the limitations of naive cosine similarity search.
3.  **Local Inference (Ollama)**: Uses `llama3.2` for reasoning and `nomic-embed-text` for high-quality technical embeddings, ensuring data privacy and low cost.
4.  **Memory**: Maintains a conversational history to allow for follow-up questions and context-aware research.
5.  **Observability**: A dedicated "Thought Process" log allows users to inspect the agent's internal reasoning for every query.

## Choice of Technologies (Justification)

-   **LangGraph**: Chosen for its explicit control over agent states and cycles. Unlike simple chains, LangGraph allows the agent to loop back for clarification or refine its search if the first retrieval is insufficient.
-   **RAG-Fusion**: Technical papers often use highly specific but varied terminology. RAG-Fusion expands the user's query into multiple perspectives, significantly improving recall for complex technical requests.
-   **Ollama (Llama 3.2)**: A lightweight yet capable model for local reasoning. `llama3.2` was chosen as a balance between speed and the ability to follow the structured instructions required for the agent loop.
-   **ChromaDB**: A simple, persistent vector store that integrates seamlessly with the LangChain ecosystem.

## Setup & Usage

### Prerequisites
- Install [Ollama](https://ollama.com/)
- Pull required models:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

### Installation
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `.\venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`

### Running the System
1. **Fetch Corpus**: `python fetch_corpus.py` (Downloads ~50-100 recent PDFs)
2. **Ingest**: `python ingest.py` (Extracts text and builds vector index)
3. **Chat**: `python cli.py` (Interactive session with observability)

## Evaluation

Run the evaluation harness to see how the agent handles technical, ambiguous, and out-of-domain questions:
```bash
python eval.py
```
The results will be saved in `eval_report.md`.

## Failure Modes & Handling

-   **Insufficient Context**: The agent is instructed to say "I don't know" if the retrieved papers do not contain the answer.
-   **Ambiguity**: If a query is too vague (e.g., "Tell me about the paper"), the agent enters a `CLARIFY` state to ask for more details.
-   **Out-of-Domain**: Requests unrelated to AI research are met with a polite refusal.
