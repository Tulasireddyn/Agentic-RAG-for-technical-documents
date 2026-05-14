import operator
from typing import Annotated, Sequence, TypedDict, Union, List

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from retriever import get_rag_fusion_retriever, retrieve_and_fuse
import os

# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: List[str]
    next_step: str
    thought_process: List[str]
    summary: str # Semantic memory of the conversation

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# 0. Summarizer Node
def summarizer(state: AgentState):
    print("---SUMMARIZING---")
    messages = state["messages"]
    summary = state.get("summary", "")
    
    if len(messages) > 6: # Summarize after 3 turns
        prompt = f"Summarize the preceding conversation so far into a concise paragraph, preserving all technical topics and papers discussed. Current summary: {summary}"
        new_summary = llm.invoke([SystemMessage(content=prompt)] + list(messages)).content
        return {"summary": new_summary, "messages": messages[-4:]} # Keep only last 2 turns in episodic memory
    return {"summary": summary}

# 1. Reasoner Node
def reasoner(state: AgentState):
    print("---REASONING---")
    messages = state["messages"]
    summary = state.get("summary", "")
    last_message = messages[-1].content
    
    system_prompt = f"""You are an expert AI Research Assistant. Your goal is to answer questions about technical cs.AI papers ONLY.
    
    CONVERSATION SUMMARY (Semantic Memory):
    {summary}
    
    STRICT DOMAIN CONSTRAINT: You only know about AI research, machine learning, and computer science papers in the provided corpus.
    If the user asks for anything else (recipes, sports, general knowledge, jokes), you MUST output 'REFUSE'.
    
    DECISION CRITERIA:
    - Clear technical question about AI: output 'RETRIEVE'.
    - Vague or ambiguous query about AI/papers: output 'CLARIFY'.
    - Non-AI/CS request: output 'REFUSE'.
    - Question about the conversation itself or ready to answer: output 'ANSWER'.
    
    Output ONLY the word: RETRIEVE, CLARIFY, REFUSE, or ANSWER."""
    
    # Simple classification
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
    decision = response.content.strip().upper()
    
    # Fallback if LLM gets chatty
    if "RETRIEVE" in decision: decision = "RETRIEVE"
    elif "CLARIFY" in decision: decision = "CLARIFY"
    elif "REFUSE" in decision: decision = "REFUSE"
    else: decision = "ANSWER"
    
    return {"next_step": decision, "thought_process": [f"Decision: {decision}"]}

# 2. Retriever Node
def retriever_node(state: AgentState):
    print("---RETRIEVING---")
    last_message = state["messages"][-1].content
    
    gen_queries, retriever = get_rag_fusion_retriever()
    docs = retrieve_and_fuse(last_message, gen_queries, retriever)
    
    context = [doc.page_content for doc in docs[:5]] # Top 5 fused docs
    
    return {
        "context": context, 
        "thought_process": [f"Retrieved {len(context)} relevant document chunks using RAG-Fusion."]
    }

# 3. Responder Node
def responder(state: AgentState):
    print("---RESPONDING---")
    context = "\n\n".join(state["context"])
    messages = state["messages"]
    
    system_prompt = f"""You are an expert AI Research Assistant. Use the following retrieved context to answer the user's question.
    If the context doesn't contain the answer, say you don't know.
    
    CONTEXT:
    {context}
    """
    
    response = llm.invoke([SystemMessage(content=system_prompt)] + list(messages))
    return {"messages": [AIMessage(content=response.content)]}

# 4. Clarifier Node
def clarifier(state: AgentState):
    print("---CLARIFYING---")
    last_message = state["messages"][-1].content
    system_prompt = "The user's query is ambiguous. Ask a polite clarifying question to understand what specific AI research or paper they are interested in."
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
    return {"messages": [AIMessage(content=response.content)]}

# 5. Refusal Node
def refusal(state: AgentState):
    print("---REFUSING---")
    return {"messages": [AIMessage(content="I'm sorry, but I can only answer questions related to AI research and technical papers within my corpus (cs.AI). Your request seems to be outside this domain.")]}

# Define Router logic
def router(state: AgentState):
    return state["next_step"]

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("summarizer", summarizer)
workflow.add_node("reasoner", reasoner)
workflow.add_node("retriever", retriever_node)
workflow.add_node("responder", responder)
workflow.add_node("clarifier", clarifier)
workflow.add_node("refusal", refusal)

workflow.set_entry_point("summarizer")
workflow.add_edge("summarizer", "reasoner")

workflow.add_conditional_edges(
    "reasoner",
    router,
    {
        "RETRIEVE": "retriever",
        "CLARIFY": "clarifier",
        "REFUSE": "refusal",
        "ANSWER": "responder"
    }
)

workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)
workflow.add_edge("clarifier", END)
workflow.add_edge("refusal", END)

app = workflow.compile()

def run_agent(query: str, history: List[BaseMessage] = None, summary: str = ""):
    if history is None:
        history = []
    
    initial_state = {
        "messages": history + [HumanMessage(content=query)],
        "context": [],
        "next_step": "",
        "thought_process": [],
        "summary": summary
    }
    
    result = app.invoke(initial_state)
    return result

if __name__ == "__main__":
    # Test
    res = run_agent("What is the state of multi-modal models in recent AI papers?")
    print("\nFINAL ANSWER:\n", res["messages"][-1].content)
