"""
Learning Recommendations Engine

Personalized content recommendations using collaborative filtering
and user performance tracking.
"""
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.models import (
    UserPerformance, RecommendedContent, LearningPath,
    ProcessedContent, User
)
from backend.core.database import get_db_session
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class RecommendationEngine:
    """
    Learning recommendation engine using collaborative filtering.
    
    Algorithm:
    1. User-based collaborative filtering (find similar users)
    2. Item-based recommendations (find similar content)
    3. Difficulty adjustment based on performance
    4. NCERT curriculum alignment
    """
    
    def __init__(self):
        self.min_interactions = 3  # Min interactions to make recommendations
        self.similarity_threshold = 0.6  # Min similarity score
        self.max_recommendations = 10
    
    def generate_recommendations(
        self,
        user_id: uuid.UUID,
        db: Session,
        content_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized recommendations for a user.
        
        Args:
            user_id: User UUID
            db: Database session
            content_type: Filter by content type
        
        Returns:
            List of recommended content with scores
        """
        # Get user's performance history
        user_history = self._get_user_history(user_id, db)
        
        if len(user_history) < self.min_interactions:
            # Cold start: recommend popular content
            return self._get_popular_content(db, content_type)
        
        # Find similar users
        similar_users = self._find_similar_users(user_id, user_history, db)
        
        # Get content recommendations from similar users
        collaborative_recs = self._collaborative_filtering(
            user_id, similar_users, db
        )
        
        # Get content-based recommendations
        content_recs = self._content_based_filtering(
            user_history, db
        )
        
        # Merge and rank recommendations
        merged_recs = self._merge_recommendations(
            collaborative_recs,
            content_recs,
            user_history
        )
        
        # Apply difficulty adjustment
        adjusted_recs = self._adjust_for_difficulty(
            merged_recs,
            user_history
        )
        
        # Store recommendations in database
        self._store_recommendations(user_id, adjusted_recs, db)
        
        return adjusted_recs[:self.max_recommendations]
    
    def _get_user_history(
        self,
        user_id: uuid.UUID,
        db: Session
    ) -> List[Dict[str, Any]]:
        """Get user's interaction history."""
        history = db.query(UserPerformance).filter(
            UserPerformance.user_id == user_id
        ).order_by(desc(UserPerformance.created_at)).limit(100).all()
        
        return [
            {
                "content_id": str(h.content_id),
                "content_type": h.content_type,
                "score": h.score or 0.0,
                "completed": h.completed,
                "time_spent": h.time_spent or 0,
                "difficulty": h.difficulty_rating or 1
            }
            for h in history
        ]
    
    def _find_similar_users(
        self,
        user_id: uuid.UUID,
        user_history: List[Dict[str, Any]],
        db: Session
    ) -> List[tuple]:
        """
        Find users with similar interaction patterns.
        
        Returns:
            List of (user_id, similarity_score) tuples
        """
        user_content_ids = {h["content_id"] for h in user_history}
        
        # Get all other users who interacted with same content
        other_users = db.query(
            UserPerformance.user_id,
            func.count(UserPerformance.content_id).label('overlap')
        ).filter(
            UserPerformance.content_id.in_(user_content_ids),
            UserPerformance.user_id != user_id
        ).group_by(UserPerformance.user_id).all()
        
        # Calculate Jaccard similarity
        similar_users = []
        for other_user_id, overlap in other_users:
            # Get other user's content
            other_content = db.query(UserPerformance.content_id).filter(
                UserPerformance.user_id == other_user_id
            ).all()
            other_content_ids = {str(c[0]) for c in other_content}
            
            # Jaccard similarity
            union_size = len(user_content_ids | other_content_ids)
            similarity = overlap / union_size if union_size > 0 else 0
            
            if similarity >= self.similarity_threshold:
                similar_users.append((other_user_id, similarity))
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)
        
        return similar_users[:10]  # Top 10 similar users
    
    def _collaborative_filtering(
        self,
        user_id: uuid.UUID,
        similar_users: List[tuple],
        db: Session
    ) -> List[Dict[str, Any]]:
        """Get recommendations from similar users."""
        if not similar_users:
            return []
        
        user_ids = [u[0] for u in similar_users]
        similarity_map = {u[0]: u[1] for u in similar_users}
        
        # Get content consumed by similar users
        similar_user_content = db.query(UserPerformance).filter(
            UserPerformance.user_id.in_(user_ids)
        ).all()
        
        # Get user's consumed content
        user_consumed = db.query(UserPerformance.content_id).filter(
            UserPerformance.user_id == user_id
        ).all()
        user_consumed_ids = {str(c[0]) for c in user_consumed}
        
        # Score recommendations
        recommendations = defaultdict(lambda: {"score": 0.0, "count": 0})
        
        for performance in similar_user_content:
            content_id = str(performance.content_id)
            
            # Skip already consumed
            if content_id in user_consumed_ids:
                continue
            
            # Weight by user similarity
            user_similarity = similarity_map.get(performance.user_id, 0)
            
            # Weight by performance score
            performance_score = performance.score or 0.5
            
            weighted_score = user_similarity * performance_score
            recommendations[content_id]["score"] += weighted_score
            recommendations[content_id]["count"] += 1
        
        # Normalize and format
        result = []
        for content_id, data in recommendations.items():
            result.append({
                "content_id": content_id,
                "score": data["score"] / data["count"],
                "algorithm": "collaborative_filtering"
            })
        
        return sorted(result, key=lambda x: x["score"], reverse=True)
    
    def _content_based_filtering(
        self,
        user_history: List[Dict[str, Any]],
        db: Session
    ) -> List[Dict[str, Any]]:
        """Recommend similar content based on what user liked."""
        # Get high-scoring content
        liked_content_ids = [
            h["content_id"] for h in user_history
            if h["score"] >= 0.7 or h["completed"]
        ]
        
        if not liked_content_ids:
            return []
        
        # Get metadata for liked content
        liked_content = db.query(ProcessedContent).filter(
            ProcessedContent.id.in_(liked_content_ids)
        ).all()
        
        # Extract tags/categories
        tags = set()
        for content in liked_content:
            if content.metadata:
                tags.update(content.metadata.get("tags", []))
        
        # Find content with similar tags
        similar_content = db.query(ProcessedContent).filter(
            ProcessedContent.id.notin_(liked_content_ids),
            ProcessedContent.metadata.op('?|')(list(tags))  # JSONB contains any key
        ).limit(50).all()
        
        # Score by tag overlap
        result = []
        for content in similar_content:
            content_tags = set(content.metadata.get("tags", []))
            overlap = len(tags & content_tags)
            score = overlap / len(tags) if tags else 0
            
            if score > 0:
                result.append({
                    "content_id": str(content.id),
                    "score": score,
                    "algorithm": "content_based"
                })
        
        return sorted(result, key=lambda x: x["score"], reverse=True)
    
    def _merge_recommendations(
        self,
        collaborative: List[Dict[str, Any]],
        content_based: List[Dict[str, Any]],
        user_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge recommendations from different algorithms."""
        merged = {}
        
        # Weight: 60% collaborative, 40% content-based
        for rec in collaborative:
            content_id = rec["content_id"]
            merged[content_id] = {
                "content_id": content_id,
                "score": rec["score"] * 0.6,
                "algorithms": ["collaborative_filtering"]
            }
        
        for rec in content_based:
            content_id = rec["content_id"]
            if content_id in merged:
                merged[content_id]["score"] += rec["score"] * 0.4
                merged[content_id]["algorithms"].append("content_based")
            else:
                merged[content_id] = {
                    "content_id": content_id,
                    "score": rec["score"] * 0.4,
                    "algorithms": ["content_based"]
                }
        
        return sorted(
            merged.values(),
            key=lambda x: x["score"],
            reverse=True
        )
    
    def _adjust_for_difficulty(
        self,
        recommendations: List[Dict[str, Any]],
        user_history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Adjust recommendations based on user's difficulty level."""
        # Calculate user's average performance
        avg_score = sum(h["score"] for h in user_history) / len(user_history)
        avg_difficulty = sum(h["difficulty"] for h in user_history) / len(user_history)
        
        # Determine optimal difficulty
        if avg_score >= 0.8:
            # High performer: recommend harder content
            target_difficulty = min(avg_difficulty + 1, 5)
        elif avg_score < 0.5:
            # Struggling: recommend easier content
            target_difficulty = max(avg_difficulty - 1, 1)
        else:
            # Maintain current level
            target_difficulty = avg_difficulty
        
        # Boost scores for content at target difficulty
        for rec in recommendations:
            rec["target_difficulty"] = target_difficulty
            rec["reason"] = f"Based on your {avg_score:.1%} success rate"
        
        return recommendations
    
    def _store_recommendations(
        self,
        user_id: uuid.UUID,
        recommendations: List[Dict[str, Any]],
        db: Session
    ):
        """Store recommendations in database."""
        for rec in recommendations:
            existing = db.query(RecommendedContent).filter(
                RecommendedContent.user_id == user_id,
                RecommendedContent.content_id == uuid.UUID(rec["content_id"])
            ).first()
            
            if existing:
                # Update score
                existing.recommendation_score = rec["score"]
                existing.created_at = datetime.now(timezone.utc)
            else:
                # Create new
                db.add(RecommendedContent(
                    id=uuid.uuid4(),
                    user_id=user_id,
                    content_id=uuid.UUID(rec["content_id"]),
                    content_type="lesson",
                    recommendation_score=rec["score"],
                    recommendation_reason=rec.get("reason"),
                    algorithm=",".join(rec["algorithms"])
                ))
        
        db.commit()
    
    def _get_popular_content(
        self,
        db: Session,
        content_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get popular content for cold start."""
        query = db.query(
            UserPerformance.content_id,
            func.count(UserPerformance.id).label('interaction_count'),
            func.avg(UserPerformance.score).label('avg_score')
        ).group_by(UserPerformance.content_id)
        
        if content_type:
            query = query.filter(UserPerformance.content_type == content_type)
        
        popular = query.order_by(
            desc('interaction_count')
        ).limit(self.max_recommendations).all()
        
        return [
            {
                "content_id": str(p[0]),
                "score": p[2] or 0.5,
                "algorithm": "popular",
                "reason": f"Popular content with {p[1]} interactions"
            }
            for p in popular
        ]
    
    def track_interaction(
        self,
        user_id: uuid.UUID,
        content_id: uuid.UUID,
        interaction_type: str,
        score: Optional[float] = None,
        time_spent: Optional[int] = None,
        completed: bool = False,
        difficulty_rating: Optional[int] = None,
        db: Session = None
    ):
        """Track user interaction with content."""
        if not db:
            db = get_db_session()
        
        performance = UserPerformance(
            id=uuid.uuid4(),
            user_id=user_id,
            content_id=content_id,
            content_type="lesson",
            interaction_type=interaction_type,
            score=score,
            time_spent=time_spent,
            completed=completed,
            difficulty_rating=difficulty_rating
        )
        
        db.add(performance)
        db.commit()
        
        logger.info(f"Tracked interaction: user={user_id}, content={content_id}, type={interaction_type}")
