import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

class FeedbackStore:
    def __init__(self, feedback_file: str = "feedback_data.json"):
        self.feedback_file = Path(feedback_file)
        self._ensure_feedback_file()
    
    def _ensure_feedback_file(self) -> None:
        """Create feedback file if it doesn't exist"""
        if not self.feedback_file.exists():
            self.feedback_file.write_text(json.dumps({"feedbacks": {}}))
    
    def _load_feedbacks(self) -> Dict:
        """Load feedback data from file"""
        try:
            return json.loads(self.feedback_file.read_text())
        except Exception:
            return {"feedbacks": {}}
    
    def _save_feedbacks(self, data: Dict) -> None:
        """Save feedback data to file"""
        self.feedback_file.write_text(json.dumps(data, indent=2))
    
    def add_feedback(self, question: str, original_answer: str, feedback: str) -> None:
        """Add new feedback for a question"""
        data = self._load_feedbacks()
        
        # Create normalized key for the question
        question_key = question.lower().strip()
        
        # Add new feedback entry
        feedback_entry = {
            "original_answer": original_answer,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "verified": False  # For potential future moderation
        }
        
        # Update feedbacks
        data["feedbacks"][question_key] = feedback_entry
        self._save_feedbacks(data)
    
    def get_feedback(self, question: str) -> Optional[Dict]:
        """Get feedback for a specific question if exists"""
        data = self._load_feedbacks()
        question_key = question.lower().strip()
        return data["feedbacks"].get(question_key)
    
    def search_similar_feedbacks(self, question: str, similarity_threshold: float = 0.8) -> List[Dict]:
        """Find similar questions in feedback store"""
        from difflib import SequenceMatcher
        
        data = self._load_feedbacks()
        similar_feedbacks = []
        
        for stored_question, feedback in data["feedbacks"].items():
            similarity = SequenceMatcher(None, question.lower(), stored_question).ratio()
            if similarity >= similarity_threshold:
                similar_feedbacks.append({
                    "question": stored_question,
                    "feedback": feedback,
                    "similarity": similarity
                })
        
        return sorted(similar_feedbacks, key=lambda x: x["similarity"], reverse=True)