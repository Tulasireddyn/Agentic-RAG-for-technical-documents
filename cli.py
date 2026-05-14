from agent import run_agent
from langchain_core.messages import HumanMessage, AIMessage
import sys

def chat():
    print("=== Agentic RAG CLI (Ollama) ===")
    print("Type 'exit' or 'quit' to stop.\n")
    
    history = []
    summary = ""
    
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        print("\n--- Agent Thinking ---")
        try:
            result = run_agent(query, history, summary)
            
            # Observability: Print thought process
            for thought in result.get("thought_process", []):
                print(f"[DEBUG] {thought}")
            
            new_summary = result.get("summary", "")
            if new_summary != summary:
                print(f"[MEMORY] Summary Updated: {new_summary}")
                summary = new_summary
            
            answer = result["messages"][-1].content
            print(f"\nAssistant: {answer}\n")
            
            # Update history for memory
            # The agent might have trimmed messages in its own state, 
            # so we sync back the messages if they were returned.
            history = result["messages"]
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat()
