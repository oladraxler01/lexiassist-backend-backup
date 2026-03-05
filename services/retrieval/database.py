from sqlalchemy import create_engine, Column, String, Integer, Text, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List, Dict
import os

# Try to import pgvector
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    print("⚠️  pgvector not available - will use mock data")

Base = declarative_base()

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    material_id = Column(String, index=True)
    user_id = Column(String, index=True)
    chunk_text = Column(Text)
    embedding = Vector(384) if PGVECTOR_AVAILABLE else Column(Text)
    chunk_index = Column(Integer)

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lexiassist:password@localhost:5432/lexiassist_db"
)

engine = None
SessionLocal = None

if PGVECTOR_AVAILABLE:
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        print("✅ Retrieval database connection configured (will connect when used)")
    except Exception as e:
        print(f"⚠️  Database setup error: {e}")
        SessionLocal = None

def search_similar_chunks(query_vector: List[float], user_id: str, material_id: str = None, top_k: int = 5) -> List[Dict]:
    """
    STEP 3: Find chunks with vectors most similar to query vector
    Uses pgvector's distance operators
    """

    # FORCE MOCK MODE for now (until docker-compose arrives)
    # Remove this "if True" when PostgreSQL is running
    if True or not PGVECTOR_AVAILABLE or not SessionLocal:
        print("   ⚠️  Using mock search (PostgreSQL not connected)")
        print(f"   Would search for user: {user_id}, material: {material_id}")
        return [
            {
                "chunk_id": "mock-1",
                "material_id": material_id or "ibm-hackathon-test",
                "chunk_text": f"This is a mock result for query. When PostgreSQL is ready, I'll search {top_k} chunks.",
                "similarity_score": 0.95,
                "chunk_index": 0
            },
            {
                "chunk_id": "mock-2",
                "material_id": material_id or "ibm-hackathon-test",
                "chunk_text": "This is mock result 2. Real search will use pgvector's <=> operator to compare the 384-dim vectors.",
                "similarity_score": 0.87,
                "chunk_index": 1
            },
            {
                "chunk_id": "mock-3",
                "material_id": material_id or "ibm-hackathon-test",
                "chunk_text": f"Query vector first 3 values: {query_vector[:3]}...",
                "similarity_score": 0.82,
                "chunk_index": 2
            }
        ]

    # REAL MODE: This will work when docker-compose starts PostgreSQL
    try:
        db = SessionLocal()

        # Build the query with user security filter
        query = select(DocumentChunk).where(DocumentChunk.user_id == user_id)

        # Optional: filter by specific material
        if material_id:
            query = query.where(DocumentChunk.material_id == material_id)

        # THE MAGIC: Order by vector similarity
        # Using cosine distance (can use .cosine_distance() method instead of .op())
        query = query.order_by(DocumentChunk.embedding.cosine_distance(query_vector)).limit(top_k)

        results = db.execute(query).scalars().all()

        return [
            {
                "chunk_id": r.id,
                "material_id": r.material_id,
                "chunk_text": r.chunk_text,
                "similarity_score": 1.0 - r.embedding.cosine_distance(query_vector),
                "chunk_index": r.chunk_index
            }
            for r in results
        ]

    except Exception as e:
        print(f"   ⚠️  Search error: {e}")
        print("   Falling back to mock results...")
        return [
            {
                "chunk_id": "error-fallback",
                "material_id": material_id or "default",
                "chunk_text": f"Search failed: {str(e)}. Using fallback.",
                "similarity_score": 0.0,
                "chunk_index": 0
            }
        ]
    finally:
        if 'db' in locals():
            db.close()
