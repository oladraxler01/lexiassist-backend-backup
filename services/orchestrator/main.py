from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import google.generativeai as genai
import os
import uvicorn
from datetime import datetime
import uuid

# Initialize FastAPI
app = FastAPI(
    title="LexiAssist AI Orchestrator",
    description="Manages Gemini AI calls, prompts, and conversation history",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API (placeholder - will use env variable in production)
# For testing without API key, we'll use mock responses
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    print("✅ Gemini AI configured")
else:
    print("⚠️  No GEMINI_API_KEY - using mock mode")
    model = None

# Conversation history store (temporary - use Redis/DB in production)
conversation_history: Dict[str, List[Dict]] = {}

# Pydantic models
class ChatRequest(BaseModel):
    query: str = Field(..., description="User's question")
    user_id: str = Field(..., description="For conversation history")
    material_id: Optional[str] = Field(None, description="Specific document context")
    context_chunks: List[str] = Field(default=[], description="Retrieved text chunks from Retrieval Service")
    conversation_id: Optional[str] = Field(None, description="Continue existing conversation")

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    tokens_used: int
    model: str
    sources: List[str]  # Which chunks were used

class ContextItem(BaseModel):
    text: str
    source: str
    relevance_score: float

# Health check
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "ai-orchestrator",
        "port": 5005,
        "version": "1.0.0",
        "ai_model": "gemini-pro" if model else "mock-mode",
        "api_key_configured": GEMINI_API_KEY is not None
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "gemini-pro" if model else "mock-mode",
        "conversations_active": len(conversation_history)
    }

def build_prompt(query: str, context_chunks: List[str], chat_history: List[Dict]) -> str:
    """
    Build a smart prompt for Gemini with context and history
    """
    # Build context section
    context_text = "\n\n".join([
        f"[Document {i+1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks[:5])  # Top 5 chunks
    ]) if context_chunks else "No specific document context provided."

    # Build conversation history
    history_text = ""
    if chat_history:
        history_text = "\n\nPrevious conversation:\n"
        for msg in chat_history[-3:]:  # Last 3 messages
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    # Full prompt
    prompt = f"""You are LexiAssist, a helpful AI study assistant. Answer the user's question based on the provided document context.

DOCUMENT CONTEXT:
{context_text}
{history_text}

USER QUESTION: {query}

INSTRUCTIONS:
- Answer based ONLY on the document context provided
- If the answer isn't in the context, say "I don't have enough information in the provided documents"
- Be concise but thorough
- Use bullet points for lists
- Cite which document you're referencing [Document X]

Your response:"""

    return prompt

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - receives query + context, returns AI response
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Get or create conversation history
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []

    chat_history = conversation_history[conversation_id]

    # Build the prompt
    prompt = build_prompt(request.query, request.context_chunks, chat_history)

    print(f"\n🤖 Processing chat for user {request.user_id}")
    print(f"   Query: {request.query[:50]}...")
    print(f"   Context chunks: {len(request.context_chunks)}")
    print(f"   History length: {len(chat_history)}")

    try:
        if model:
            # REAL GEMINI CALL
            response = model.generate_content(prompt)
            ai_response = response.text
            tokens_used = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
        else:
            # MOCK MODE (for testing without API key)
            ai_response = f"[MOCK RESPONSE] Based on the {len(request.context_chunks)} document chunks provided, I would answer: {request.query}\n\nNote: This is a mock response. Set GEMINI_API_KEY environment variable for real AI."
            tokens_used = 0

        # Update conversation history
        chat_history.append({"role": "user", "content": request.query, "timestamp": datetime.now().isoformat()})
        chat_history.append({"role": "assistant", "content": ai_response, "timestamp": datetime.now().isoformat()})

        # Keep only last 10 messages to prevent memory bloat
        conversation_history[conversation_id] = chat_history[-10:]

        print(f"   ✅ Response generated ({len(ai_response)} chars)")

        return ChatResponse(
            response=ai_response,
            conversation_id=conversation_id,
            tokens_used=tokens_used,
            model="gemini-pro" if model else "mock-mode",
            sources=[f"chunk_{i}" for i in range(len(request.context_chunks))]
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

@app.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """
    Retrieve conversation history
    """
    if conversation_id not in conversation_history:
        return {"conversation_id": conversation_id, "messages": []}

    return {
        "conversation_id": conversation_id,
        "messages": conversation_history[conversation_id]
    }

@app.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """
    Clear conversation history
    """
    if conversation_id in conversation_history:
        del conversation_history[conversation_id]
        return {"message": "Conversation cleared"}

    return {"message": "Conversation not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005, reload=True)
