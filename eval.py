from agent import run_agent
from langchain_core.messages import HumanMessage, AIMessage

def run_evaluation():
    test_cases = [
        {
            "id": 1,
            "query": "What are the latest advancements in multi-modal LLMs according to the recent papers?",
            "expected": "Technical answer based on retrieved context."
        },
        {
            "id": 2,
            "query": "How is Reinforcement Learning being used in AI agents in 2026?",
            "expected": "Technical answer based on retrieved context."
        },
        {
            "id": 3,
            "query": "Summarize the key findings on prompt engineering from the corpus.",
            "expected": "Summary based on retrieved context."
        },
        {
            "id": 4,
            "query": "What is the impact of Quantization on small language models in these papers?",
            "expected": "Technical answer based on retrieved context."
        },
        {
            "id": 5,
            "query": "Are there any papers discussing ethical implications of AI agents?",
            "expected": "Identification of relevant ethical discussions."
        },
        {
            "id": 6,
            "query": "What are the common benchmarks used for evaluating AI agents in this corpus?",
            "expected": "List of benchmarks mentioned in context."
        },
        {
            "id": 7,
            "query": "Tell me about the paper.",
            "expected": "CLARIFICATION: Agent should ask which paper."
        },
        {
            "id": 8,
            "query": "How does it work?",
            "expected": "CLARIFICATION: Agent should ask what 'it' refers to."
        },
        {
            "id": 9,
            "query": "What is the best recipe for chocolate cake?",
            "expected": "REFUSAL: Agent should state it only knows about AI research."
        },
        {
            "id": 10,
            "query": "Who won the World Series in 2025?",
            "expected": "REFUSAL: Agent should state it only knows about AI research."
        }
    ]
    
    print("=== STARTING EVALUATION ===\n")
    results = []
    
    for case in test_cases:
        print(f"Test Case {case['id']}: {case['query']}")
        try:
            res = run_agent(case['query'])
            answer = res["messages"][-1].content
            decision = res["next_step"]
            
            print(f"Agent Decision: {decision}")
            print(f"Agent Answer: {answer[:200]}...")
            print("-" * 30)
            
            results.append({
                "id": case['id'],
                "query": case['query'],
                "decision": decision,
                "answer": answer
            })
        except Exception as e:
            print(f"Error in test case {case['id']}: {e}")
            
    # Generate a simple report
    with open("eval_report.md", "w") as f:
        f.write("# Evaluation Report\n\n")
        f.write("| ID | Query | Decision | Answer (Snippet) |\n")
        f.write("|----|-------|----------|------------------|\n")
        for r in results:
            snippet = r['answer'][:100].replace("\n", " ")
            f.write(f"| {r['id']} | {r['query']} | {r['decision']} | {snippet}... |\n")
            
    print("\nEvaluation complete. Report saved to eval_report.md")

if __name__ == "__main__":
    run_evaluation()
