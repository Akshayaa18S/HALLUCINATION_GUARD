"""
Result model for storing final analysis results
"""
from sqlalchemy import Column, String, DateTime, Float, Text, ForeignKey, JSON
from datetime import datetime
import uuid

from .base import Base


class Result(Base):
    """
    Result model for storing final analysis results
    
    Attributes:
        id: Unique identifier
        job_id: Reference to parent job
        hallucination_score: Score 0-100
        confidence: Confidence 0-1
        is_hallucination: "yes", "no", or "uncertain"
        generated_response: LLM-generated response
        hidden_states: Extracted hidden states
        extracted_features: Extracted features from pipeline
        retrieved_evidence: Retrieved evidence from RAG
        supporting_documents: Supporting documents
        contradictions: Contradicting evidence
        verified_answer: Corrected answer
        shap_explanation: SHAP explainability
        important_tokens: Important tokens identified
        attention_heatmap: Path to attention heatmap image
        explanation_text: Human-readable explanation
        total_processing_time_ms: Total execution time
        created_at: When result was created
        updated_at: When result was last updated
    """
    
    __tablename__ = "results"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("jobs.job_id"), unique=True, nullable=False)
    user_query = Column(Text, nullable=True)
    execution_pipeline = Column(JSON, nullable=True)
    
    # Hallucination Detection Results
    hallucination_score = Column(Float, nullable=True)  # 0-100
    confidence = Column(Float, nullable=True)  # 0-1
    is_hallucination = Column(String, nullable=True)  # "yes", "no", "uncertain"
    
    # Generated Response
    generated_response = Column(Text, nullable=True)
    
    # Hidden States & Features (stored as JSON)
    hidden_states = Column(JSON, nullable=True)
    extracted_features = Column(JSON, nullable=True)
    
    # RAG Verification
    retrieved_evidence = Column(JSON, nullable=True)
    supporting_documents = Column(JSON, nullable=True)
    contradictions = Column(JSON, nullable=True)
    verified_answer = Column(Text, nullable=True)
    
    # Explainability
    shap_explanation = Column(JSON, nullable=True)
    important_tokens = Column(JSON, nullable=True)
    attention_heatmap = Column(String, nullable=True)  # Path to image
    explanation_text = Column(Text, nullable=True)
    
    # Processing
    total_processing_time_ms = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<Result for Job {self.job_id}>"
