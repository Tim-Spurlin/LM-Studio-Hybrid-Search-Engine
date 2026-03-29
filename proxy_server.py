import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import re
import httpx
import h5py
import faiss
import GPUtil
import numpy as np
import uuid
import asyncio
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import CrossEncoder
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from contextlib import asynccontextmanager

from imdds_truth_filter import IMDDS_TruthFilter

app = FastAPI()
imdds = IMDDS_TruthFilter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "tauri://localhost", "http://localhost:8081", "http://localhost:1420", "http://tauri.localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HDF5_PATH = os.environ.get("HDF5_PATH", "./universal_knowledge_base.h5")
LM_STUDIO_URL = "http://127.0.0.1:8888/v1"
LM_STUDIO_HEADERS = {
    "Authorization": "Bearer your_api_key_here"
}
API_KEY = os.environ.get("LM_STUDIO_API_KEY", "your_api_key_here")
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
import glob

# Global ML State
embedder = None
cross_encoder = None
cached_faiss_index = None
cached_bm25_index = None
cached_db_rows = 0
cached_chunks_raw = []

# Increase chunk size for personal/biographical documents
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256

# === Q8_0 NEMOTRON EMBED 1B — GPU-ONLY PRODUCTION BLOCK (2026-03-17) ===
from llama_cpp import Llama
import GPUtil

print("=== EMBEDDER START ===")

model_path = os.environ.get("EMBED_MODEL_PATH", "./models/llama-nemotron-embed-1b-v2-fixed.gguf")

# DYNAMIC GPU VRAM SCALING for RTX 3050 Ti (4GB)
free_mb = GPUtil.getGPUs()[0].memoryFree
# Nemotron Q8_0 needs ~1300MB total for 16 layers (~81MB/layer). Reserve 250MB for context.
available_for_layers = free_mb - 250
if available_for_layers <= 0:
    n_gpu_layers = 0
else:
    n_gpu_layers = min(16, int(available_for_layers / 85))

print(f"VRAM safety: {free_mb:.1f} MB free → dynamically scaled to {n_gpu_layers}/16 GPU layers to prevent CUDA OOM")

import llama_cpp
embedder = Llama(
    model_path=model_path,
    embedding=True,
    n_ctx=8192,
    n_gpu_layers=n_gpu_layers,
    n_batch=512,
    n_threads=8,
    verbose=False,
    chat_format=None,
    pooling_type=llama_cpp.LLAMA_POOLING_TYPE_MEAN
)

import math
from typing import List
import numpy as np

def matryoshka_slice_and_normalize(raw_emb: List[float]) -> List[float]:
    TARGET_DIM = 768
    vec = np.array(raw_emb[:TARGET_DIM], dtype=np.float32)
    if len(vec) < TARGET_DIM:
        raise ValueError(f"Embedding dim {len(raw_emb)} is less than target {TARGET_DIM}.")
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        raise ValueError("Zero vector produced — model may have failed silently.")
    normalized = (vec / norm).tolist()
    final_norm = np.linalg.norm(normalized)
    assert abs(final_norm - 1.0) < 1e-5, f"Normalization failed: final norm={final_norm}"
    return normalized

def get_embedding(text: str):
    raw_emb = embedder.embed(text)
    return matryoshka_slice_and_normalize(raw_emb)

from functools import lru_cache
import hashlib

# Cache up to 512 query embeddings in memory
@lru_cache(maxsize=512)
def get_embedding_cached(text: str) -> tuple:
    """
    Cached version of get_embedding.
    Returns a tuple (hashable for lru_cache) instead of a list.
    """
    result = get_embedding(text)
    return tuple(result)

def get_embedding_as_array(text: str) -> list:
    """
    Public interface — returns list as before but hits cache first.
    """
    return list(get_embedding_cached(text))

_embed_semaphore = asyncio.Semaphore(1)

async def embed_queries_parallel(queries: list) -> np.ndarray:
    """
    Embeds multiple queries with thread-safe serialized GPU access.
    The semaphore prevents concurrent llama_cpp calls which are not thread-safe.
    """
    async def safe_embed(q):
        async with _embed_semaphore:
            return await asyncio.to_thread(get_embedding_as_array, q)
    results = await asyncio.gather(*[safe_embed(q) for q in queries])
    return np.array(results)

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256
print("=== EMBEDDER LOADED SUCCESSFULLY — Q8_0 ON GPU ===")
print("=== EMBEDDER INITIALIZATION COMPLETE — SYSTEM ONLINE ===")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cross_encoder
    print(f"[Lifespan] Booting Proxy. Native Embedder routing to Llama-CPP GGUF Engine")
    try:
        print(f"[Lifespan] Loading CrossEncoder: {CROSS_ENCODER_NAME}...")
        import torch
        _ce_device = "cuda" if torch.cuda.is_available() else "cpu"
        cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device=_ce_device)
        print(f"[CrossEncoder] Loaded on {_ce_device}")
        print("[Startup] Pre-building adaptive strategy prototype embeddings...")
        adaptive_strategy._build_prototype_embeddings()
        print("[Startup] Prototype embeddings ready — zero cold-start latency on first query")
    except Exception as e:
        print(f"[Lifespan] Failed to load CrossEncoder: {e}")
    yield

app = FastAPI(lifespan=lifespan)

HNSW_INDEX_PATH = os.environ.get("HNSW_INDEX_PATH", "./faiss_hnsw.index")

def build_indexes(embeddings, texts):
    global cached_faiss_index, cached_bm25_index, cached_chunks_raw
    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]

    cached_chunks_raw = texts

    # Try loading persisted index first
    if os.path.exists(HNSW_INDEX_PATH):
        try:
            loaded = faiss.read_index(HNSW_INDEX_PATH)
            if loaded.ntotal == num_vectors:
                cached_faiss_index = loaded
                cached_faiss_index.hnsw.efSearch = 64
                print(f"[HNSW] Loaded persisted index ({num_vectors} vectors) — skipped rebuild")
                return
            else:
                print(f"[HNSW] Persisted index has {loaded.ntotal} vectors, DB has {num_vectors} — rebuilding")
        except Exception as e:
            print(f"[HNSW] Failed to load persisted index: {e} — rebuilding")

    # Build fresh only when necessary
    print(f"[HNSW] Building new index over {num_vectors} vectors...")
    faiss.normalize_L2(embeddings)

    M = 32
    ef_construction = 200

    index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = 64

    index.add(embeddings)
    cached_faiss_index = index

    # Persist immediately so next restart is instant
    faiss.write_index(index, HNSW_INDEX_PATH)
    print(f"[HNSW] Built and persisted index to {HNSW_INDEX_PATH}")

def rrf_score(faiss_indices, bm25_scores, k=60):
    """Reciprocal Rank Fusion (RRF) algorithm"""
    scores = {}
    
    # FAISS scores
    for rank, doc_id in enumerate(faiss_indices):
        if doc_id == -1: continue
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    # BM25 scores
    # Sort the numpy array by indices
    bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
    for rank, doc_id in enumerate(bm25_ranked_indices):
        if bm25_scores[doc_id] <= 0: break # Skip irrelevant documents
        # Only care about top N BM25 results to match FAISS length roughly
        if rank > len(faiss_indices) * 2: break
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Return sorted (doc_id, score)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidates: list,
    final_k: int,
    lambda_param: float = 0.6
) -> list:
    """
    Selects chunks that are simultaneously:
    - Relevant to the query (high cosine with query)
    - Diverse from already selected chunks (low cosine with selections)

    lambda_param: 0 = maximum diversity, 1 = maximum relevance
    0.6 is the optimal balance for RAG contexts
    """
    if len(candidates) <= final_k:
        return candidates

    selected_indices = []
    remaining_indices = list(range(len(candidates)))

    query_emb = query_embedding.reshape(1, -1)

    for _ in range(final_k):
        if not remaining_indices:
            break

        best_idx = None
        best_score = -np.inf

        for idx in remaining_indices:
            relevance = cosine_similarity(
                candidate_embeddings[idx].reshape(1, -1),
                query_emb
            )[0][0]

            if selected_indices:
                selected_embs = candidate_embeddings[selected_indices]
                redundancy = np.max(cosine_similarity(
                    candidate_embeddings[idx].reshape(1, -1),
                    selected_embs
                ))
            else:
                redundancy = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]

def contextual_compression(chunk, query):
    """Semantic context compression via sliding window similarity to preserve coherent narrative."""
    sentences = [s.strip() for s in chunk.split('.') if s.strip()]
    if not sentences:
        return chunk.strip()
        
    try:
        q_emb = np.array([get_embedding_as_array(query)])
        sent_embs = np.array([get_embedding_as_array(s) for s in sentences])
        scores = cosine_similarity(q_emb, sent_embs)[0]
        
        keep = set()
        similarity_threshold = 0.35
        window_size = 3
        
        for i, score in enumerate(scores):
            if score >= similarity_threshold:
                for j in range(max(0, i - window_size), min(len(sentences), i + window_size + 1)):
                    keep.add(j)
                    
        if not keep:
            return chunk.strip()
            
        preserved = [sentences[i] for i in sorted(keep)]
        return '. '.join(preserved) + '.'
    except Exception as e:
        print(f"Semantic compression failed, falling back to full chunk: {e}")
        return chunk.strip()


class AdaptiveRetrievalStrategy:
    """
    Uses semantic similarity against intent prototypes to determine
    retrieval depth. No hardcoded keywords. No category lists.
    Works for any domain automatically.
    """

    # These are not keyword lists — they are semantic anchor descriptions
    # that define retrieval BEHAVIOR, not subject matter.
    # You never need to update these as you add new knowledge domains.
    INTENT_PROTOTYPES = {
        "exhaustive": [
            "list everything about a person's background",
            "give me the complete history of all events",
            "what are all the positions and roles someone has held",
            "enumerate every item in a collection",
            "tell me everything stored about this topic",
        ],
        "precise": [
            "what is the exact definition of this term",
            "what is the specific value or measurement",
            "when did this single event occur",
            "what is the correct formula or equation",
            "give me the exact answer to this question",
        ],
        "synthesis": [
            "explain how multiple concepts connect together",
            "compare and contrast these approaches",
            "what are the implications of combining these ideas",
            "analyze the relationship between these systems",
            "summarize the key themes across this subject",
        ],
        "exploratory": [
            "what do you know about this broad topic",
            "give me an overview of this field",
            "what are the main ideas in this domain",
            "introduce me to this subject",
            "what should I know about this area",
        ],
    }

    # Retrieval parameters per intent — behavior not subject
    STRATEGY_PARAMS = {
        "exhaustive": {"faiss_k": 100, "rerank_k": 50, "final_k": 25, "ef_search": 128},
        "synthesis":  {"faiss_k": 70,  "rerank_k": 35, "final_k": 18, "ef_search": 64},
        "exploratory":{"faiss_k": 60,  "rerank_k": 25, "final_k": 12, "ef_search": 48},
        "precise":    {"faiss_k": 30,  "rerank_k": 15, "final_k": 6,  "ef_search": 32},
    }

    def __init__(self):
        self._prototype_embeddings = None

    def _build_prototype_embeddings(self):
        """
        Lazily builds prototype embeddings on first use.
        Uses your existing GGUF embedder — no new models needed.
        """
        if self._prototype_embeddings is not None:
            return

        self._prototype_embeddings = {}
        for intent, phrases in self.INTENT_PROTOTYPES.items():
            phrase_embs = np.array([get_embedding_as_array(p) for p in phrases])
            # Average the phrase embeddings into a single intent centroid
            centroid = np.mean(phrase_embs, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            self._prototype_embeddings[intent] = centroid

    def classify(self, query: str) -> dict:
        """
        Classifies query intent by semantic similarity.
        Returns retrieval parameters tuned to that intent.
        No keywords. No domain knowledge required.
        """
        self._build_prototype_embeddings()

        query_emb = np.array(get_embedding_as_array(query)).reshape(1, -1)

        best_intent = "exploratory"
        best_score = -1.0

        for intent, centroid in self._prototype_embeddings.items():
            score = cosine_similarity(query_emb, centroid.reshape(1, -1))[0][0]
            if score > best_score:
                best_score = score
                best_intent = intent

        params = self.STRATEGY_PARAMS[best_intent].copy()
        params["intent"] = best_intent
        params["confidence"] = round(float(best_score), 3)

        print(f"[AdaptiveStrategy] '{query[:60]}...' → {best_intent} "
              f"(confidence={params['confidence']}) "
              f"FAISS:{params['faiss_k']} Rerank:{params['rerank_k']} Final:{params['final_k']}")

        return params


# Initialize once at module level — reused across all requests
adaptive_strategy = AdaptiveRetrievalStrategy()

async def semantic_query_expansion(query: str, top_neighbor_chunks: list) -> list[str]:
    """
    Generates expansion queries by analyzing what the nearest
    semantic neighbors actually contain — no LLM call needed.
    Faster and domain-agnostic.
    """
    if not top_neighbor_chunks:
        return [query]

    expansion_queries = [query]

    # Extract the first 2 neighbor chunks as semantic context
    # and ask the LLM to generate targeted follow-up queries
    # based on what gaps exist between query and neighbors
    context_sample = "\n".join([c[:200] for c in top_neighbor_chunks[:2]])

    prompt = f"""Query: {query}

Closest database content found:
{context_sample}

Generate 2 alternative search queries that would find related 
information NOT already covered by the above content.
Return exactly 2 lines, one query per line, nothing else."""

    # This is a much smaller, targeted LLM call than before
    # because it has concrete context to work from
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{LM_STUDIO_URL}/chat/completions",
                json={
                    "model": "local-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 60
                },
                headers=LM_STUDIO_HEADERS
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                new_queries = [
                    q.strip() for q in content.split("\n")
                    if q.strip() and len(q.strip()) > 5
                ]
                expansion_queries.extend(new_queries[:2])
    except Exception:
        pass

    return expansion_queries


# ======== Endpoints ========

import sys
from typing import TypedDict, Annotated, List, Any
import operator
from langgraph.graph import StateGraph, END

# ======== LangGraph State Definition ========

class RAGState(TypedDict):
    original_question: str
    current_queries: List[str]
    accumulated_context: Annotated[List[str], operator.add]
    iteration_count: int
    ready_to_answer: bool
    final_answer: Any
    messages: List[dict] # Original chat history
    model: str

# ======== LangGraph Nodes ========

async def formulate_queries_node(state: RAGState):
    """Generates initial or follow-up search queries via semantic neighbor mapping."""
    global cached_faiss_index, cached_chunks_raw
    question = state["original_question"]
    iteration = state["iteration_count"]
    
    print(f"\n[LangGraph Phase 1] Formulating queries for iteration {iteration}...")
    
    # Fast mini-FAISS to get top 2 chunks for semantic expansion
    top_neighbor_chunks = []
    if cached_faiss_index is not None and len(cached_chunks_raw) > 0:
        try:
            q_emb_vec = await asyncio.to_thread(get_embedding_as_array, question)
            q_emb = np.array([q_emb_vec], dtype=np.float32)
            faiss.normalize_L2(q_emb)
            faiss_res = await asyncio.to_thread(cached_faiss_index.search, q_emb, 2)
            _, faiss_indices = faiss_res
            top_neighbor_chunks = [cached_chunks_raw[i] for i in faiss_indices[0] if i != -1][:2]
        except Exception as e:
            print(f"[LangGraph] Mini-FAISS neighbor extraction failed: {e}")

    queries = await semantic_query_expansion(question, top_neighbor_chunks)
    
    print(f" -> Generated queries: {queries}")
    return {"current_queries": queries[:4]}

async def retrieve_and_compress_node(state: RAGState):
    """Hits HDF5 database with current queries and compresses the best chunks."""
    global cached_db_rows, cached_faiss_index, cached_bm25_index
    queries = state["current_queries"]
    print(f"\n[LangGraph Phase 2] Retrieving data via Hybrid Search...")
    
    if not os.path.exists(HDF5_PATH):
        print(" -> HDF5 database not found.")
        return {"accumulated_context": []}

    new_context = []
    try:
        with h5py.File(HDF5_PATH, 'r', swmr=True) as f:
            if 'chunks/embeddings' in f and 'chunks/text' in f:
                embeddings_ds = f['chunks/embeddings']
                text_ds = f['chunks/text']
                
                if embeddings_ds.shape[0] > 0:
                    num_chunks = embeddings_ds.shape[0]
                    
                    if cached_faiss_index is None or cached_db_rows != num_chunks:
                        print(f"[Index] Rebuilding — chunk count changed {cached_db_rows} → {num_chunks}")
                        db_texts = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in text_ds[:]]
                        db_embeddings = np.array(embeddings_ds[:]).reshape(-1, 768)
                        build_indexes(db_embeddings, db_texts)
                        cached_db_rows = num_chunks
                        del db_embeddings  # Free ~9GB RAM immediately
                    
                    # Direct Llama-CPP Embedding Inference
                    raw_embs_array = await embed_queries_parallel(queries)
                            
                    avg_query_embedding = np.mean(raw_embs_array, axis=0, keepdims=True)
                    faiss.normalize_L2(avg_query_embedding)
                    
                    combined_query_text = " ".join(queries)
                    tokenized_query = combined_query_text.lower().split(" ")
                    
                    # Run Semantic FAISS Search Asynchronously
                    strategy = adaptive_strategy.classify(state["original_question"])
                    
                    if hasattr(cached_faiss_index, 'hnsw'):
                        cached_faiss_index.hnsw.efSearch = strategy["ef_search"]
                    
                    faiss_res = await asyncio.to_thread(cached_faiss_index.search, avg_query_embedding, min(strategy["faiss_k"], num_chunks))
                    
                    _, faiss_indices = faiss_res
                    top_rrf_ids = [doc_id for doc_id in faiss_indices[0] if doc_id != -1][:strategy["faiss_k"]]
                    
                    candidate_texts = [cached_chunks_raw[i] for i in top_rrf_ids]
                    retrieved_chunks = [{'text': t, 'embedding': np.zeros(768, dtype=np.float32), 'title': 'Vector Document'} for t in candidate_texts]
                    
                    # === UNIVERSAL FULL-FIDELITY RETRIEVAL — No query type restrictions ===
                    # All queries receive identical full retrieval treatment.
                    # IMDDS truth scoring is preserved as metadata for transparency but never
                    # used to gate or reduce the candidate pool before cross-encoder ranking.
                    # This ensures biographical, technical, medical, and all other query types
                    # receive the same maximum retrieval quality without name-specific bypasses.
                    all_scored = imdds.filter_and_score(retrieved_chunks)
                    all_scored.sort(key=lambda x: x['truth_score'], reverse=True)

                    # Preserve high-truth chunks at the front but keep ALL chunks in the pool
                    # so the cross-encoder — not the truth filter — makes the final ranking decision
                    high_truth = [s for s in all_scored if s['truth_score'] >= 0.5]
                    low_truth = [s for s in all_scored if s['truth_score'] < 0.5]
                    filtered_scored = high_truth + low_truth  # Cross-encoder sees everything, sorted by truth score
                    
                    top_imdds_chunks = filtered_scored # Feed all retrieved chunks to Cross-Encoder
                    
                    cross_inp = [[state["original_question"], s['chunk']['text']] for s in top_imdds_chunks]
                    if len(cross_inp) > 0:
                        cross_scores = cross_encoder.predict(cross_inp)
                        scored_candidates = sorted(zip(cross_scores, top_imdds_chunks), key=lambda x: x[0], reverse=True)
                        
                        combined_query_text = " ".join(queries).lower()
                        is_biomed = any(kw in combined_query_text for kw in ['disease', 'drug', 'cure', 'treatment', 'medication', 'pathway', 'compound'])
                        
                        hyp_text = ""
                        if is_biomed:
                            high_truth = [s for s in filtered_scored if s['truth_score'] > imdds.config.get('truth_score_threshold', 0.75)]
                            if high_truth:
                                hypothesis = imdds.generate_biomed_hypothesis(high_truth)
                                hyp_text = f"[MATHEMATICAL BIOMEDICAL HYPOTHESIS via SVD + Kalman]\n{hypothesis}\n\n"
                                
                        # MMR Deduplication — skip re-embedding, use cross-encoder rank order as diversity proxy
                        text_to_fscore = {f_score['chunk']['text']: f_score for _, f_score in scored_candidates}
                        diverse_chunks = []
                        seen_prefixes = set()
                        for _, f_score in scored_candidates:
                            text = f_score['chunk']['text']
                            prefix = text[:200]  # Use first 200 chars as dedup fingerprint
                            if prefix not in seen_prefixes:
                                seen_prefixes.add(prefix)
                                diverse_chunks.append(text)
                            if len(diverse_chunks) >= strategy["final_k"]:
                                break
                                
                        for text in diverse_chunks: 
                            f_score = text_to_fscore[text]
                            truth_tag = f"[Truth Score: {f_score['truth_score']:.3f} | {f_score['explanation']}]\n"
                            
                            final_chunk_str = hyp_text + truth_tag + text.strip()
                            hyp_text = "" # Only prepend hypothesis once
                            new_context.append(final_chunk_str)
                        
    except Exception as e:
        print(f" -> Retrieval failed: {e}")
        
    print(f" -> Extracted {len(new_context)} new compressed contexts.")
    return {"accumulated_context": new_context}

async def evaluate_context_node(state: RAGState):
    """The Detective Brain: Checks if we have the answer, or if we need to loop."""
    question = state["original_question"]
    context_so_far = "\n---\n".join(state["accumulated_context"])
    iteration = state["iteration_count"]
    
    print(f"\n[LangGraph Phase 3] Evaluating context (Iteration {iteration})...")
    
    # 1. Ask the LLM to critique the current state
    critique_prompt = f"""You are a scientific detective connecting the dots.
Original Question: {question}

Context found so far:
{context_so_far}

Task: Evaluate if the context provided is SUFFICIENT to fully and accurately answer the question.
If it is sufficient, reply with ONLY the word "SUFFICIENT".
If it is missing crucial specific details, reply with a short sentence describing exactly what specific piece of information is still missing. Do not answer the question yet."""

    payload = {
        "model": state.get("model", "local-model"),
        "messages": [{"role": "user", "content": critique_prompt}],
        "temperature": 0.1,
        "max_tokens": 150
    }
    
    needs_loop = True
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(f"{LM_STUDIO_URL}/chat/completions", json=payload, headers=LM_STUDIO_HEADERS)
            if response.status_code == 200:
                critique = response.json().get("choices", [])[0].get("message", {}).get("content", "").strip()
                print(f" -> Assessment: {critique}")
                if "SUFFICIENT" in critique.upper():
                    needs_loop = False
    except Exception as e:
        print(f" -> Assessment failed: {e}")
        needs_loop = False # Fallback to answering
        
    # Prevent infinite loops
    if iteration >= 2: 
        print(" -> Max iterations reached. Forcing resolution.")
        needs_loop = False
        
    if needs_loop:
        return {"iteration_count": iteration + 1}
    
    # 2. If Sufficient (or max iterations), Flag Ready
    print("\n[LangGraph Phase 4] Context Gathered. Ready for final streaming generation...")
    return {"ready_to_answer": True}

def router_edge(state: RAGState):
    """Decides whether to loop back to searching or to end."""
    if state.get("ready_to_answer", False):
        return "end"
    return "search"

# ======== Compile LangGraph ========

graph_builder = StateGraph(RAGState)
graph_builder.add_node("formulate", formulate_queries_node)
graph_builder.add_node("retrieve", retrieve_and_compress_node)
graph_builder.add_node("evaluate", evaluate_context_node)

graph_builder.set_entry_point("formulate")
graph_builder.add_edge("formulate", "retrieve")
graph_builder.add_edge("retrieve", "evaluate")
graph_builder.add_conditional_edges(
    "evaluate", 
    router_edge,
    {"search": "formulate", "end": END}
)

rag_app = graph_builder.compile()

# ======== Endpoints ========

@app.post("/embed")
async def generate_embedding(request: Request):
    """Endpoint for Rust DropWatcher to get MPNet embeddings."""
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {API_KEY}":
        return {"error": "Unauthorized"}

    data = await request.json()
    texts = data.get("texts", [])
    if not texts or embedder is None:
        return {"embeddings": []}
    
    # Route directly to local GGUF engine
    try:
        # Offload the heavy blocking loop to a detached thread so FastAPI doesn't freeze!
        embeddings = await asyncio.to_thread(lambda texts: [get_embedding_as_array(t) for t in texts], texts)
        return {"embeddings": embeddings}
    except Exception as e:
        print(f"[Embed Proxy] Embedding array failure: {e}")
        return {"embeddings": []}

class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    domain_filter: Optional[str] = None

@app.post("/search")
async def search_database(req: SearchRequest):
    """Fast, single-pass hybrid search for the LM Studio UI Plugin."""
    global cached_db_rows, cached_faiss_index, cached_bm25_index
    
    if not os.path.exists(HDF5_PATH):
        return {"results": []}
        
    try:
        with h5py.File(HDF5_PATH, 'r', swmr=True) as f:
            if 'chunks/embeddings' not in f or 'chunks/text' not in f:
                return {"results": []}

            embeddings_ds = f['chunks/embeddings']
            text_ds = f['chunks/text']
            
            if embeddings_ds.shape[0] == 0:
                return {"results": []}
                
            num_chunks = embeddings_ds.shape[0]
                
            # Rebuild index if DB grew
            if cached_faiss_index is None or cached_db_rows != num_chunks:
                print(f"[Search] Rebuilding — chunk count changed {cached_db_rows} → {num_chunks}")
                db_texts = [t.decode('utf-8') for t in text_ds[:]]
                db_embeddings = np.array(embeddings_ds[:]).reshape(-1, 768)
                build_indexes(db_embeddings, db_texts)
                cached_db_rows = num_chunks
                del db_embeddings  # Free ~9GB RAM immediately
                
            strategy = adaptive_strategy.classify(req.query)
            FAISS_RECALL_K = strategy["faiss_k"]
            RERANK_CANDIDATE_K = strategy["rerank_k"]
            FINAL_CONTEXT_K = strategy["final_k"]
            
            if hasattr(cached_faiss_index, 'hnsw'):
                cached_faiss_index.hnsw.efSearch = strategy["ef_search"]
            
            # 1. Asynchronous FAISS Search using Local GGUF
            q_emb_vec = await asyncio.to_thread(get_embedding_as_array, req.query)
            q_emb = np.array([q_emb_vec], dtype=np.float32)
            faiss.normalize_L2(q_emb)
            
            faiss_res = await asyncio.to_thread(cached_faiss_index.search, q_emb, min(FAISS_RECALL_K, num_chunks))
            _, faiss_indices = faiss_res
            
            # 3. Fetch Contexts (Broad Recall)
            top_rrf_ids = [doc_id for doc_id in faiss_indices[0] if doc_id != -1][:FAISS_RECALL_K]
            candidate_texts = [cached_chunks_raw[i] for i in top_rrf_ids]
            
            # 4. Cross-Encoder Re-Ranking (Tiered: all FAISS retrieved chunks)
            pruned_candidates = candidate_texts # Cross-Encoder ranks all FAISS candidates
            cross_inp = [[req.query, text] for text in pruned_candidates]
            cross_scores = await asyncio.to_thread(cross_encoder.predict, cross_inp)
            scored = sorted(zip(cross_scores, pruned_candidates), key=lambda x: x[0], reverse=True)
            
            # 5. Return Final Context Window to LLM
            top_contexts = [text for score, text in scored[:FINAL_CONTEXT_K]]
            return {"results": top_contexts}
            
    except Exception as e:
        print(f"Search endpoint failed: {e}")
        return {"results": []}

class LearnRequest(BaseModel):
    text: str

@app.post("/learn")
async def learn_from_chat(req: LearnRequest):
    """Saves LM Studio chat prompts and attachments directly into the KnowledgeDrop folder for permanent OfflineRAG ingestion."""
    try:
        if not req.text.strip():
            return {"status": "ignored"}
            
        drop_folder = os.environ.get("KNOWLEDGE_DROP", "./KnowledgeDrop")
        if not os.path.exists(drop_folder):
            os.makedirs(drop_folder, exist_ok=True)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"LMStudio_Chat_{timestamp}_{unique_id}.txt"
        
        filepath = os.path.join(drop_folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(req.text)
            
        print(f"[Learn] Successfully dropped learning feedback: {filename}")
        return {"status": "success", "file": filename}
    except Exception as e:
        print(f"[Learn] Failed to save learning feedback: {e}")
        return {"status": "error", "detail": str(e)}

@app.get("/stats")
async def get_stats():
    """Returns the macro metrics of the neural database for the Brain UI."""
    try:
        size = os.path.getsize(HDF5_PATH) if os.path.exists(HDF5_PATH) else 0
        return {"vectors": cached_db_rows, "memory_bytes": size}
    except Exception as e:
        return {"vectors": 0, "memory_bytes": 0, "error": str(e)}

@app.get("/history")
async def get_history():
    """Retrieves all unique ingested files for the Live Feed UI."""
    try:
        with h5py.File(HDF5_PATH, "r") as f:
            if "chunks/metadata" in f:
                dset = f["chunks/metadata"]
                total = len(dset)
                data = dset[:]
                res = []
                seen = set()
                for item in reversed(data):
                    source_file = item["source_file"].decode('utf-8') if isinstance(item["source_file"], bytes) else item["source_file"]
                    if source_file not in seen:
                        seen.add(source_file)
                        res.append({
                            "source_file": source_file,
                            "domain": item["domain"].decode('utf-8') if isinstance(item["domain"], bytes) else item["domain"],
                            "timestamp": int(item["timestamp"]),
                        })
                        if len(res) >= 500: # reasonable safety cap
                            break
                return {"history": res}
    except Exception as e:
        print(f"History Fetch error: {e}")
    return {"history": []}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Advanced Iterative RAG Pipeline proxying to LM Studio."""
    auth_header = request.headers.get("Authorization")
    if auth_header != f"Bearer {API_KEY}":
        return {"error": "Unauthorized"}

    data = await request.json()
    messages = data.get("messages", [])
    
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break
            
    if not user_query:
        return {"error": "No user query found."}
        
    if data.get("skip_rag", False):
        import httpx
        from fastapi.responses import StreamingResponse
        # Hardcode specific model for bypass operations
        data["model"] = "liquid/lfm2.5-1.2b"
        data.pop("skip_rag", None)  # Remove non-standard parameter before forwarding
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            if data.get("stream", False):
                async def stream_gen():
                    async with client.stream("POST", f"{LM_STUDIO_URL}/chat/completions", json=data, headers=LM_STUDIO_HEADERS) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                return StreamingResponse(stream_gen(), media_type="text/event-stream")
            else:
                resp = await client.post(f"{LM_STUDIO_URL}/chat/completions", json=data, headers=LM_STUDIO_HEADERS, timeout=300.0)
                return resp.json()

    # [Conversational Memory Drop] - Save every prompt and attachment to the KnowledgeDrop!
    try:
        import time, uuid, os
        chat_log_dir = os.environ.get("CHAT_LOGS_DIR", "./KnowledgeDrop/Chat_Logs")
        os.makedirs(chat_log_dir, exist_ok=True)
        timestamp = int(time.time())
        file_path = f"{chat_log_dir}/chat_{timestamp}_{uuid.uuid4().hex[:6]}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"USER SUBMITTED PROMPT TO LM STUDIO:\n\n{user_query}")
        print(f"[Memory Stream] Intercepted user query/attachments and dropped into {file_path}")
    except Exception as mem_e:
        print(f"[Memory Stream] Failed to log to KnowledgeDrop: {mem_e}")

    print(f"\n========== NEW GRAPH EXECUTION ==========\nTargeting Query: {user_query}")
    
    # Resolve Model ID dynamically if generic
    model_id = data.get("model", "local-model")
    if model_id in ["local-model", "", None]:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{LM_STUDIO_URL}/models", headers=LM_STUDIO_HEADERS)
                if res.status_code == 200:
                    models_data = res.json().get("data", [])
                    # Pick the text generation model, not the embedding one
                    for m in models_data:
                        if "nomic" not in m["id"] and "embed" not in m["id"]:
                            model_id = m["id"]
                            break
        except Exception:
            pass

    # Initialize LangGraph State
    initial_state = {
        "original_question": user_query,
        "current_queries": [],
        "accumulated_context": [],
        "iteration_count": 0,
        "ready_to_answer": False,
        "final_answer": None,
        "messages": messages,
        "model": model_id
    }
    
    # Run the State Machine Iterator
    try:
        final_state = await rag_app.ainvoke(initial_state)
    except Exception as e:
        print(f"LangGraph execution failed: {e}")
        return {"error": f"Internal graph error: {str(e)}"}
        
    # Generate final answer via StreamingResponse
    context_so_far = "\n---\n".join(final_state.get("accumulated_context", []))
    final_messages = list(messages)
    sys_prompts = [m for m in final_messages if m.get("role") == "system"]
    
    if sys_prompts:
        sys_prompts[0]["content"] += f"\n\nContext information gathered through research:\n{context_so_far}"
    else:
        final_messages.insert(0, {
            "role": "system",
            "content": f"You are a helpful AI assistant. Use the following carefully researched context to answer the user's question.\n\nContext:\n{context_so_far}"
        })
        
    final_payload = {
        "model": model_id,
        "messages": final_messages,
        "stream": True 
    }
    
    from fastapi.responses import StreamingResponse
    import httpx
    
    async def stream_generator():
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                async with client.stream("POST", f"{LM_STUDIO_URL}/chat/completions", json=final_payload, headers=LM_STUDIO_HEADERS) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except Exception as e:
                print(f"Streaming error: {e}")
                
    return StreamingResponse(stream_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

