import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, loads

def get_rag_fusion_retriever(db_dir="db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # LLM for query generation
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    # Multi-query prompt
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user query, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    generate_queries = (
        prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    )
    
    return generate_queries, retriever

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of retrieved documents 
        and an optional parameter k used in the RRF formula """
    
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return only the documents
    return [doc for doc, score in reranked_results]

def retrieve_and_fuse(question, generate_queries, retriever):
    # 1. Generate multi-queries
    queries = generate_queries.invoke({"question": question})
    print(f"Generated queries: {queries}")
    
    # 2. Retrieve for each query
    all_docs = []
    for q in queries:
        all_docs.append(retriever.invoke(q))
    
    # 3. Fuse results
    fused_docs = reciprocal_rank_fusion(all_docs)
    return fused_docs
