"""Question generation service using Llama for educational content."""
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid

from sqlalchemy import Column, String, Integer, Text, JSON, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session

from ..core.config import settings
from ..utils.logging import get_logger
from ..core.database import get_db_session, Base
from ..utils.model_loader import ModelLoader

logger = get_logger(__name__)


class GeneratedQuestion(Base):
    """Database model for AI-generated questions."""
    
    __tablename__ = "generated_questions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("processed_content.id"), nullable=False)
    question_text = Column(Text, nullable=False)
    question_type = Column(String(50), nullable=False)  # mcq, short_answer, true_false
    options = Column(JSON, nullable=True)  # For MCQ questions
    correct_answer = Column(Text, nullable=False)
    explanation = Column(Text, nullable=True)
    difficulty = Column(String(20), nullable=False)  # easy, medium, hard
    ncert_objective = Column(String(200), nullable=True)
    bloom_taxonomy_level = Column(String(50), nullable=True)  # remember, understand, apply, analyze
    quality_score = Column(Integer, default=0)  # 0-100, manual review score
    is_approved = Column(Integer, default=0)  # 0=pending, 1=approved, -1=rejected
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    question_metadata = Column('metadata', JSON, default=dict)  # Renamed from 'metadata' to avoid SQLAlchemy reserved word


class QuestionGenerator:
    """
    Generate educational questions from documents using Llama-3.2-3B.
    
    Features:
    - Multiple question types (MCQ, short answer, true/false)
    - NCERT learning objective tagging
    - Bloom's taxonomy classification
    - Quality scoring and review workflow
    """
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        
        # Prompt templates
        self.prompts = {
            "mcq": """Generate a multiple choice question based on this text.

Text: {text}

Generate a question with:
1. Clear question statement
2. 4 options (A, B, C, D)
3. Correct answer
4. Brief explanation

Format your response as JSON:
{{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "A",
    "explanation": "..."
}}""",
            
            "short_answer": """Generate a short answer question based on this text.

Text: {text}

Generate a question that requires a 2-3 sentence answer.

Format your response as JSON:
{{
    "question": "...",
    "correct_answer": "...",
    "explanation": "..."
}}""",
            
            "true_false": """Generate a true/false question based on this text.

Text: {text}

Generate a statement that can be evaluated as true or false.

Format your response as JSON:
{{
    "question": "...",
    "correct_answer": "true" or "false",
    "explanation": "..."
}}"""
        }
    
    async def generate_questions(
        self,
        document_id: str,
        text: str,
        num_questions: int = 5,
        question_types: Optional[List[str]] = None,
        ncert_objective: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate questions from document text.
        
        Args:
            document_id: Document UUID
            text: Document text content
            num_questions: Number of questions to generate (5-10)
            question_types: Types to generate (mcq, short_answer, true_false)
            ncert_objective: NCERT learning objective to tag
            
        Returns:
            List of generated question dictionaries
        """
        if question_types is None:
            question_types = ["mcq", "mcq", "mcq", "short_answer", "true_false"]
        
        # Ensure we don't exceed limits
        num_questions = min(num_questions, 10)
        
        # Split text into chunks for context
        chunks = self._chunk_text(text, max_length=500)
        
        questions = []
        for i, qtype in enumerate(question_types[:num_questions]):
            # Select relevant chunk
            chunk = chunks[i % len(chunks)]
            
            # Generate question
            question_data = await self._generate_single_question(
                chunk, qtype, ncert_objective
            )
            
            if question_data:
                question_data["document_id"] = document_id
                questions.append(question_data)
        
        # Save to database
        with get_db_session() as session:
            for q_data in questions:
                question = GeneratedQuestion(
                    document_id=uuid.UUID(document_id),
                    question_text=q_data["question"],
                    question_type=q_data["type"],
                    options=q_data.get("options"),
                    correct_answer=q_data["correct_answer"],
                    explanation=q_data.get("explanation"),
                    difficulty=q_data.get("difficulty", "medium"),
                    ncert_objective=ncert_objective,
                    bloom_taxonomy_level=q_data.get("bloom_level", "understand"),
                    quality_score=0,
                    is_approved=0
                )
                session.add(question)
            
            session.commit()
            logger.info(f"Generated {len(questions)} questions for document {document_id}")
        
        return questions
    
    async def _generate_single_question(
        self,
        text: str,
        question_type: str,
        ncert_objective: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Generate a single question using Llama."""
        try:
            # Get prompt template
            prompt = self.prompts[question_type].format(text=text)
            
            # Load model (lazy loading)
            model, tokenizer = self.model_loader.load_causal_lm_model(self.model_name)
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON from response
            question_data = self._parse_response(response, question_type)
            
            if question_data:
                question_data["type"] = question_type
                question_data["difficulty"] = self._estimate_difficulty(text)
                question_data["bloom_level"] = self._classify_bloom_taxonomy(question_data["question"])
                
                return question_data
            
        except Exception as e:
            logger.error(f"Question generation failed: {e}", exc_info=True)
        
        return None
    
    def _parse_response(self, response: str, question_type: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from model."""
        try:
            # Extract JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.error(f"Failed to parse question response: {e}")
        
        return None
    
    def _chunk_text(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into chunks."""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks or [text[:max_length]]
    
    def _estimate_difficulty(self, text: str) -> str:
        """Estimate question difficulty based on text complexity."""
        words = text.split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        
        if avg_word_length < 5:
            return "easy"
        elif avg_word_length < 7:
            return "medium"
        else:
            return "hard"
    
    def _classify_bloom_taxonomy(self, question: str) -> str:
        """Classify question using Bloom's taxonomy."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["what", "who", "when", "list", "define"]):
            return "remember"
        elif any(word in question_lower for word in ["explain", "describe", "summarize"]):
            return "understand"
        elif any(word in question_lower for word in ["apply", "use", "demonstrate"]):
            return "apply"
        elif any(word in question_lower for word in ["analyze", "compare", "contrast"]):
            return "analyze"
        elif any(word in question_lower for word in ["evaluate", "judge", "critique"]):
            return "evaluate"
        else:
            return "create"
    
    def approve_question(self, question_id: str, score: int = 100):
        """Approve a generated question."""
        with get_db_session() as session:
            question = session.query(GeneratedQuestion).filter_by(id=uuid.UUID(question_id)).first()
            if question:
                question.is_approved = 1
                question.quality_score = score
                session.commit()
                logger.info(f"Approved question {question_id} with score {score}")
    
    def get_questions_for_review(self, limit: int = 20) -> List[GeneratedQuestion]:
        """Get questions pending review."""
        with get_db_session() as session:
            return session.query(GeneratedQuestion).filter_by(is_approved=0).limit(limit).all()


# Global generator instance
question_generator = QuestionGenerator()
