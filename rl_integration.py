"""
Integration layer between RL Controller and Flask app
"""

import sys
sys.path.append('./src')

from controller import PostApplyController

class RLIntegration:
    """Wrapper for RL controller with Flask-friendly API"""
    
    def __init__(self):
        self.controller = PostApplyController()
        print("✅ RL Controller loaded successfully!")
    
    def add_job(self, company, role, description, job_url=None):
        """Add job application and get initial analysis"""
        result = self.controller.add_application(
            company=company,
            role=role,
            description=description,
            job_url=job_url
        )
        return result
    
    def get_strategy(self, application_id):
        """Get complete strategy for application"""
        recommendations = self.controller.get_recommendations(application_id)
        return recommendations
    
    def score_message(self, application_id, message_text):
        """Score a follow-up message"""
        scores = self.controller.score_message(application_id, message_text)
        return scores

# Test it
if __name__ == "__main__":
    rl = RLIntegration()
    print("✅ RL Integration ready!")
