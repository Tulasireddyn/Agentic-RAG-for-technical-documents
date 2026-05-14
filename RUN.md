# Running the Agentic RAG Project

Follow these steps to start the backend and frontend.

### 1. Prerequisites
- Ensure **Ollama** is running.
- Ensure you have pulled the models:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

### 2. Activate Environment
In the root directory:
```powershell
.\venv\Scripts\activate
```

### 3. Start the Backend API
In the root directory:
```powershell
python api.py
```
The API will run on `http://localhost:8000`.

### 4. Start the Frontend
In a **new terminal** window, go to the `frontend` directory:
```powershell
cd frontend
npm run dev
```
The frontend will usually run on `http://localhost:5173`. Open this URL in your browser.

### 5. Ingestion (If needed)
If you haven't ingested any papers yet:
```powershell
python fetch_corpus.py
python ingest.py
```

## Features
- **Thought Trace**: Watch the agent's internal reasoning (Summarizing -> Reasoning -> Retrieving -> Responding) in the sidebar.
- **Semantic Memory**: View the current conversation summary stored in the agent's long-term memory.
- **Modern UI**: Dark mode with glassmorphism and smooth transitions.
