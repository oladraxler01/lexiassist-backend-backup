from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import os
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="LexiAssist Evaluation Service",
    description="Analytics, quiz grading, and feedback collection",
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

# Pydantic models for Analytics (Phase 4)
class QuizSubmission(BaseModel):
    quiz_id: str
    user_id: str
    answers: Dict[str, Any]  # {question_id: answer}
    time_taken_seconds: int

class GradeResponse(BaseModel):
    attempt_id: str
    quiz_id: str
    user_id: str
    score: float  # Percentage or points
    correct_answers: Dict[str, Any]
    feedback: Dict[str, str]  # Per-question feedback

class AIInteractionLog(BaseModel):
    user_id: str
    service_type: str = Field(..., description="e.g., 'chat', 'summary'")
    input_tokens: int
    output_tokens: int
    latency_ms: int
    success: bool
    model_name: Optional[str] = "gemini-pro"

class FeedbackSubmission(BaseModel):
    interaction_id: Optional[str] = None
    user_id: str
    rating: int = Field(..., ge=1, le=5, description="1-5 star rating")
    comment: Optional[str] = None
    feature_type: str = Field(..., description="e.g., 'chat_response', 'quiz_hint'")

class AnalyticsResponse(BaseModel):
    total_interactions: int
    average_latency_ms: float
    total_tokens_consumed: int
    success_rate: float

# Health check
@app.get("/")
async def root():
    return {"status": "healthy", "service": "evaluation", "port": 8083}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "database": "connected",
        "services": ["grading", "analytics", "feedback"]
    }

# TODO: Phase 4 Step 5 - Quiz Auto-Marking
@app.post("/grade-quiz", response_model=GradeResponse)
async def grade_quiz(submission: QuizSubmission):
    """
    Auto-grades quiz submissions.
    Called by Frontend when student submits quiz.

    TODO:
    1. Fetch correct answers from Content Service (Team Member 2) or DB
    2. Compare submission.answers with correct answers
    3. Calculate score
    4. Save to quiz_attempts table
    5. Return grade to frontend
    """

    # Placeholder grading logic (implement real comparison)
    correct_count = 0
    total_questions = len(submission.answers)

    # TODO: Replace with actual answer key lookup
    correct_answers = {}  # Fetch from Content Service

    for q_id, user_answer in submission.answers.items():
        # Placeholder comparison
        is_correct = False  # Compare with correct_answers[q_id]
        if is_correct:
            correct_count += 1

    score = (correct_count / total_questions * 100) if total_questions > 0 else 0

    return GradeResponse(
        attempt_id="uuid-placeholder",
        quiz_id=submission.quiz_id,
        user_id=submission.user_id,
        score=score,
        correct_answers=correct_answers,
        feedback={}  # Add per-question feedback logic
    )

# TODO: Phase 4 Step 4 - Analytics Tracking
@app.post("/log-interaction")
async def log_ai_interaction(log: AIInteractionLog):
    """
    Logs AI usage metrics (tokens, latency, costs).
    Called by AI/ML Service after each Gemini call.

    TODO: Save to ai_interactions table (SQLAlchemy model)
    """

    # Calculate approximate cost (example)
    cost_per_1k_tokens = 0.0005  # Adjust based on model
    total_tokens = log.input_tokens + log.output_tokens
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens

    print(f"Logging interaction for user {log.user_id}: {total_tokens} tokens")

    # TODO: Insert into database
    # db_session.add(ai_interaction_obj)
    # db_session.commit()

    return {
        "status": "logged",
        "interaction_id": "uuid-placeholder",
        "estimated_cost_usd": estimated_cost
    }

# TODO: Phase 4 Step 4 - Feedback Collection
@app.post("/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """
    Collects user ratings (1-5) and comments on AI responses.
    Called by Frontend after user rates a chat response.

    TODO: Save to feedback table
    """

    print(f"Feedback received: {feedback.rating}/5 stars from user {feedback.user_id}")

    # TODO: Insert into feedback table
    # db_session.add(feedback_obj)
    # db_session.commit()

    return {
        "status": "saved",
        "feedback_id": "uuid-placeholder",
        "thank_you_message": "Thanks for your feedback!"
    }

# Analytics Dashboard Endpoints
@app.get("/analytics/{user_id}")
async def get_user_analytics(user_id: str):
    """
    Returns study analytics for a specific user.
    TODO: Query ai_interactions and quiz_attempts tables
    """
    return {
        "user_id": user_id,
        "total_study_time_minutes": 0,
        "quizzes_completed": 0,
        "average_quiz_score": 0,
        "ai_interactions_today": 0
    }

@app.get("/analytics/system/summary")
async def get_system_analytics():
    """
    Admin dashboard: System-wide analytics.
    TODO: Aggregate all ai_interactions
    """
    return AnalyticsResponse(
        total_interactions=0,
        average_latency_ms=0,
        total_tokens_consumed=0,
        success_rate=100.0
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083, reload=True)
