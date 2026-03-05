from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from database import search_similar_chunks
import os
import uvicorn

# Import the same embedding model used in Ingestion
print("Loading embedding model for Retrieval Service... (one-time load)")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded!")

app = FastAPI(
    title="LexiAssist Retrieval Service",
    description="Semantic search & vector retrieval for RAG",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class RetrieveRequest(BaseModel):
    query: str = Field(..., description="User's question")
    user_id: str = Field(..., description="For security filtering")
    material_id: Optional[str] = Field(None, description="Optional: filter by specific material")
    top_k: int = Field(default=5, le=10, description="Number of chunks to return")

class ChunkResult(BaseModel):
    chunk_id: str
    material_id: str
    chunk_text: str
    similarity_score: float
    chunk_index: int

class RetrieveResponse(BaseModel):
    query: str
    query_embedding_preview: List[float]  # Show first 5 numbers for debugging
    results: List[ChunkResult]
    cached: bool = False
    note: str

# Health check
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "retrieval",
        "port": 5003,
        "model": "all-MiniLM-L6-v2",
        "embedding_dim": 384
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "vector_search": "placeholder (waiting for postgres)",
        "cache": "disabled (waiting for redis)"
    }

def generate_query_embedding(query: str) -> List[float]:
    """
    STEP 2: Convert user query to vector using SAME model as Ingestion
    """
    print(f"\n🔍 Generating embedding for query: '{query}'")
    embedding = model.encode(query)
    print(f"   ✓ Generated {len(embedding)}-dim vector")
    print(f"   Sample values: {embedding[0]:.6f}, {embedding[1]:.6f}, {embedding[2]:.6f}")
    return embedding.tolist()

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_context(request: RetrieveRequest):
    """
    Main RAG retrieval endpoint - NOW WITH REAL EMBEDDINGS + SEARCH STRUCTURE!
    """
    # STEP 2: Generate real embedding
    query_vector = generate_query_embedding(request.query)

    # STEP 3: Search for similar chunks (mock until PostgreSQL ready)
    results_data = search_similar_chunks(
        query_vector=query_vector,
        user_id=request.user_id,
        material_id=request.material_id,
        top_k=request.top_k
    )

    # Convert to Pydantic models
    chunk_results = [ChunkResult(**r) for r in results_data]

    return RetrieveResponse(
        query=request.query,
        query_embedding_preview=query_vector[:5],
        results=chunk_results,
        cached=False,
        note=f"Search returned {len(chunk_results)} chunks. Real pgvector search activates when PostgreSQL is ready."
    )
