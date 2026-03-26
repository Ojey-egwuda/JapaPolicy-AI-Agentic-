"""
Build the Vector Database for the UK Immigration Agentic RAG Assistant.
Run this ONCE before using the assistant.

Usage:
    python build_db.py
"""

import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List

load_dotenv()

from src.config import settings


def load_documents(data_dir: str = None) -> List[Document]:
    """
    Load all UK visa policy PDFs from the data directory.
    """
    if data_dir is None:
        data_dir = settings.data_dir
    results = []
    
    print(f"\n📂 Loading documents from '{data_dir}' directory...")
    print("=" * 60)
    
    if not os.path.isdir(data_dir):
        print(f"❌ Error: Data directory not found at '{data_dir}'")
        return results

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"⚠️ No PDF files found in {data_dir}")
        return results
    
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    for idx, filename in enumerate(pdf_files, 1):
        file_path = os.path.join(data_dir, filename)
        try:
            print(f"   [{idx}/{len(pdf_files)}] Loading: {filename}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            results.extend(docs)
            print(f"         ✅ Loaded {len(docs)} pages")
        except Exception as e:
            print(f"         ❌ Error: {e}")

    print("=" * 60)
    print(f"📊 Total pages loaded: {len(results)}")
    return results


def build_vector_database():
    """
    Loads documents, generates embeddings, builds BM25 index,
    and saves everything to a persistent ChromaDB collection.
    """
    print("\n" + "=" * 60)
    print("🏗️  UK Immigration RAG - Database Builder")
    print("=" * 60)
    
    start_time = time.time()

    # Step 1: Load Documents
    print("\n[Step 1/3] Loading PDF documents...")
    documents = load_documents()
    
    if not documents:
        print(f"❌ No documents found in '{settings.data_dir}' directory. Aborting.")
        print(f"\n💡 Tip: Place your gov.uk PDF files in the '{settings.data_dir}/' folder")
        return
    
    print(f"✅ Loaded {len(documents)} document pages.")

    # Step 2: Initialize VectorDB
    print("\n[Step 2/3] Initializing Vector Database...")
    try:
        from src.vectordb import VectorDB
        vector_db = VectorDB()
    except Exception as e:
        print(f"❌ Error initializing VectorDB: {e}")
        return
    
    print("✅ VectorDB initialized.")

    # Step 3: Process and Index Documents
    print("\n[Step 3/3] Processing documents...")
    print("   (Chunking → Embedding → Indexing → Saving)")
    print("   ⏳ This may take several minutes for large document sets...")
    
    try:
        vector_db.add_documents(documents)
    except Exception as e:
        print(f"❌ Error during document processing: {e}")
        return
    
    # Complete
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("✅ DATABASE BUILD COMPLETE")
    print("=" * 60)
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    
    # Print final stats
    try:
        stats = vector_db.get_collection_stats()
        print(f"\n📊 Database Statistics:")
        print(f"   • Collection: {stats['collection_name']}")
        print(f"   • Total chunks: {stats['total_chunks']}")
        print(f"   • Embedding model: {stats['embedding_model']}")
        print(f"   • Embedding dimensions: {stats['embedding_dimensions']}")
        print(f"   • Distance metric: {stats['distance_metric']}")
        print(f"   • Hybrid search: {'✅ Enabled' if stats['hybrid_search_enabled'] else '❌ Disabled'}")
    except Exception as e:
        print(f"⚠️ Could not retrieve final stats: {e}")
    
    print("\n💡 Next steps:")
    print("   1. Run: streamlit run streamlit_app.py")
    print("   2. Or test: python -c \"from src.app import AgenticRAGAssistant; a = AgenticRAGAssistant(); print(a.invoke('What is a Skilled Worker visa?'))\"")
    print()


if __name__ == "__main__":
    build_vector_database()