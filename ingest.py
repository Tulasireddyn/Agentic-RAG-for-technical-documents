import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
import glob
from dotenv import load_dotenv

load_dotenv()

def ingest_documents(corpus_dir="corpus", db_dir="db"):
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    pdf_files = glob.glob(os.path.join(corpus_dir, "*.pdf"))
    
    all_documents = []
    
    print(f"Processing {len(pdf_files)} PDF files...")
    for pdf_file in tqdm(pdf_files):
        try:
            loader = PyMuPDFLoader(pdf_file)
            docs = loader.load()
            
            # Add some metadata about the filename
            for doc in docs:
                doc.metadata["source"] = os.path.basename(pdf_file)
                
            all_documents.extend(docs)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            
    # Chunking
    # We use a larger chunk size for technical papers to keep context, 
    # but with significant overlap to ensure semantic continuity.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(all_documents)
    print(f"Split into {len(chunks)} chunks.")
    
    # Vector store
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    print("Vector store created and persisted.")
    return vectorstore

if __name__ == "__main__":
    ingest_documents()
