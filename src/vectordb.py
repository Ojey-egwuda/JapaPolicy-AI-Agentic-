import os
import chromadb
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class VectorDB:
    """
    Advanced vector database wrapper using ChromaDB with cosine similarity
    and hybrid search (semantic + keyword matching).
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database with cosine similarity.
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "uk_immigration_docs"
        )
        
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", 
            "sentence-transformers/all-mpnet-base-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"🔄 Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Create collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "UK Immigration RAG document collection",
                "hnsw:space": "cosine"
            },
        )

        # For hybrid search (BM25)
        self.bm25_corpus = []
        self.bm25_index = None
        self.doc_id_map = {}

        print(f"✅ Vector database initialized:")
        print(f"   • Collection: {self.collection_name}")
        print(f"   • Distance metric: Cosine Similarity")
        print(f"   • Embedding model: {self.embedding_model_name}")
        print(f"   • Embedding dimensions: {self.embedding_model.get_sentence_embedding_dimension()}")

    def add_documents(self, documents: List) -> None:
        """
        Add pre-chunked documents to the vector database with hybrid indexing.
        
        Args:
            documents: List of LangChain Document objects with page_content and metadata
        """
        print(f"\n🔄 Processing {len(documents)} pre-chunked documents...")

        all_texts = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []

        for i, doc in enumerate(documents):
            text = doc.page_content
            metadata = getattr(doc, "metadata", {}) or {}

            # Generate normalized embedding
            embedding = self.embedding_model.encode(
                [text], 
                normalize_embeddings=True,
                show_progress_bar=False
            )[0].tolist()

            chunk_id = f"doc_{i}"

            all_texts.append(text)
            all_embeddings.append(embedding)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)
            
            # Build BM25 corpus
            self.bm25_corpus.append(text)
            self.doc_id_map[i] = chunk_id

        if all_texts:
            # Add to ChromaDB
            self.collection.add(
                documents=all_texts,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                ids=all_ids,
            )
            
            print(f"🔄 Building BM25 keyword index...")
            tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            
            print(f"✅ Successfully indexed {len(all_texts)} chunks")
            print(f"   • Vector embeddings: {len(all_texts)}")
            print(f"   • BM25 keyword index: {len(self.bm25_corpus)} documents")
        else:
            print("⚠️ No chunks to add.")

    def search(
        self, 
        query: str, 
        n_results: int = 5, 
        min_similarity: float = 0.0,
        use_hybrid: bool = True
    ) -> Dict[str, Any]:
        """
        Search for similar documents using cosine similarity or hybrid search.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            use_hybrid: If True, combines semantic + keyword search
        
        Returns:
            Dictionary with documents, metadatas, and similarity scores (0-1 range)
        """
        if not query or not query.strip():
            print("⚠️ Empty query provided")
            return self._empty_response()
        
        print(f"🔍 Searching for: '{query}'")

        try:
            if use_hybrid and self.bm25_index:
                return self._hybrid_search(query, n_results, min_similarity)
            else:
                return self._semantic_search(query, n_results, min_similarity)

        except Exception as e:
            print(f"❌ Error during search: {e}")
            return self._empty_response()

    def _semantic_search(
        self, 
        query: str, 
        n_results: int, 
        min_similarity: float,
        is_hybrid_fallback: bool = False # Flag for logging
    ) -> Dict[str, Any]:
        """
        Semantic search using cosine similarity.
        """
        # 🔥 GENERATE NORMALIZED QUERY EMBEDDING
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,  # ← Critical for cosine
            show_progress_bar=False
        ).tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results * 2,  # Get extra for filtering
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results.get("documents") or not results["documents"][0]:
            print("⚠️ No results found.")
            return self._empty_response()
        
        # Unpack results
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        ids = results["ids"][0]
 
        # Convert cosine distances to similarities
        similarities = [max(0, min(1, 1 - (d / 2))) for d in dists]

        # Filter by minimum similarity
        filtered_results = []
        for doc, meta, sim, doc_id in zip(docs, metas, similarities, ids):
            if sim >= min_similarity:
                filtered_results.append((doc, meta, sim, doc_id))
        
        if not filtered_results:
            print(f"⚠️ No results above similarity threshold {min_similarity:.0%}")
            return self._empty_response(search_type="semantic")
        
        # Limit to n_results
        filtered_results = filtered_results[:n_results]
        
        # Unpack filtered results
        final_docs = [r[0] for r in filtered_results]
        final_metas = [r[1] for r in filtered_results]
        final_sims = [r[2] for r in filtered_results]
        final_ids = [r[3] for r in filtered_results]

        search_type = "semantic"
        # Store top semantic similarity for confidence scoring
        top_semantic_sim = final_sims[0] if final_sims else 0.0

        if not is_hybrid_fallback:
            print(f"✅ Found {len(final_docs)} matching chunks (semantic search)")
            self._print_results_summary(final_metas, final_sims)
        else:
            search_type = "hybrid (fallback)"
            print(f"⚠️ Hybrid search failed, falling back to {len(final_docs)} semantic results")
            self._print_results_summary(final_metas, final_sims)

        return {
            "documents": final_docs,
            "metadatas": final_metas,
            "similarities": final_sims,
            "ids": final_ids,
            "search_type": search_type,
            "top_semantic_sim": top_semantic_sim
        }

    def _hybrid_search(
        self, 
        query: str, 
        n_results: int, 
        min_similarity: float
    ) -> Dict[str, Any]:
        """
        HYBRID SEARCH: Combines semantic (cosine) + keyword (BM25) search
        Uses Reciprocal Rank Fusion (RRF) and falls back to semantic if poor.
        """
        # Semantic search 
        semantic_results = self._semantic_search(query, n_results * 2, 0.0)
        
        # Capture top semantic similarity for confidence scoring
        top_semantic_sim = semantic_results.get("top_semantic_sim", 0.0)
        
        if not semantic_results.get("documents"):
            print("⚠️ Hybrid search failed: No semantic results found.")
            return self._empty_response(top_semantic_sim=0.0)

        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        bm25_top_indices = sorted(
            range(len(bm25_scores)), 
            key=lambda i: bm25_scores[i], 
            reverse=True
        )[:n_results * 2]
        
        # Reciprocal Rank Fusion (RRF)
        k = 60
        combined_scores = {}
        
        # Score semantic results
        for rank, doc_id in enumerate(semantic_results["ids"]):
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 / (k + rank + 1))
        
        # Score BM25 results
        for rank, bm25_idx in enumerate(bm25_top_indices):
            if bm25_scores[bm25_idx] > 0:
                doc_id = self.doc_id_map.get(bm25_idx, f"doc_{bm25_idx}")
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 / (k + rank + 1))
        
        if not combined_scores:
            print("⚠️ Hybrid search (RRF) found no combined results, falling back to semantic.")
            return self._semantic_search(query, n_results, min_similarity, is_hybrid_fallback=True)

        # Get top combined results
        top_doc_ids = sorted(
            combined_scores.keys(), 
            key=combined_scores.get, 
            reverse=True
        )[:n_results * 2]

        # Fetch full documents from ChromaDB
        final_results = self.collection.get(
            ids=top_doc_ids,
            include=["documents", "metadatas"]
        )
        
        if not final_results or not final_results.get("documents"):
            print("⚠️ Hybrid search found no results, falling back to semantic.")
            return self._semantic_search(query, n_results, min_similarity, is_hybrid_fallback=True)

        # Re-compute similarities for final ranked list
        final_docs = final_results["documents"]
        final_metas = final_results["metadatas"]
        final_ids = final_results["ids"]
        
        final_sims = []
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        for doc in final_docs:
            doc_emb = self.embedding_model.encode([doc], normalize_embeddings=True)[0]
            sim = float(query_emb @ doc_emb)
            final_sims.append(max(0, min(1, sim)))
        
        # Filter by minimum similarity
        filtered_results = []
        for doc, meta, sim, doc_id in zip(final_docs, final_metas, final_sims, final_ids):
            if sim >= min_similarity:
                rrf_score = combined_scores.get(doc_id, 0)
                filtered_results.append((doc, meta, sim, rrf_score))
        
        MIN_HYBRID_RESULTS = 3
        
        if not filtered_results or len(filtered_results) < MIN_HYBRID_RESULTS:
            if not filtered_results:
                print(f"⚠️ No hybrid results above similarity threshold {min_similarity:.0%}, falling back to semantic.")
            else:
                print(f"⚠️ Hybrid search found only {len(filtered_results)} result(s) (min: {MIN_HYBRID_RESULTS}). Falling back to pure semantic search for more context.")
            
            # Re-run the semantic search with the correct similarity threshold
            return self._semantic_search(query, n_results, min_similarity, is_hybrid_fallback=True)

        # Sort by RRF score
        filtered_results.sort(key=lambda x: x[3], reverse=True)
        filtered_results = filtered_results[:n_results]
        
        # Unpack
        final_docs = [r[0] for r in filtered_results]
        final_metas = [r[1] for r in filtered_results]
        final_sims = [r[2] for r in filtered_results]
        
        print(f"✅ Found {len(final_docs)} matching chunks (hybrid search)")
        self._print_results_summary(final_metas, final_sims)

        return {
            "documents": final_docs,
            "metadatas": final_metas,
            "similarities": final_sims,
            "search_type": "hybrid",
            "top_semantic_sim": top_semantic_sim 
        }

    def _print_results_summary(self, metadatas: List[Dict], similarities: List[float]) -> None:
        """Print a summary of top search results."""
        for i, (meta, sim) in enumerate(zip(metadatas[:3], similarities[:3])):
            src = meta.get("source", "unknown").split("\\")[-1].split("/")[-1]
            page = meta.get("page", "N/A")
            
            quality = "🟢" if sim >= 0.85 else "🟡" if sim >= 0.70 else "🔴"
            
            print(f"   {quality} [{i+1}] {src} (page {page}) | similarity={sim:.3f} ({sim:.1%})")

    def _empty_response(self, search_type="none", top_semantic_sim=0.0) -> Dict[str, Any]:
        """Return empty response structure."""
        return {
            "documents": [],
            "metadatas": [],
            "similarities": [],
            "search_type": search_type,
            "top_semantic_sim": top_semantic_sim
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "embedding_dimensions": self.embedding_model.get_sentence_embedding_dimension(),
            "distance_metric": "cosine",
            "hybrid_search_enabled": self.bm25_index is not None
        }
