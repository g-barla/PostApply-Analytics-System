"""
Message Coach Chain
Analyzes and improves follow-up messages using RAG + RL style recommendations
"""

import json
import requests
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advanced_rag_system import AdvancedRAGSystem


class MessageCoachChain:
    """
    Intelligent message coaching that combines:
    - RAG message style best practices
    - RL Thompson Sampling style recommendations
    - Specific feedback and improvements
    """
    
    def __init__(self):
        self.rag = AdvancedRAGSystem()
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # RL Thompson Sampling success rates (from your take-home)
        self.style_performance = {
            "formal": {
                "enterprise": 0.417,
                "midsize": 0.358,
                "startup": 0.283
            },
            "casual": {
                "enterprise": 0.267,
                "midsize": 0.408,
                "startup": 0.733
            },
            "connection_focused": {
                "enterprise": 0.550,
                "midsize": 0.625,
                "startup": 0.700
            }
        }
    
    def get_rl_style_recommendation(self, company_type, has_connection):
        """Get RL-based style recommendation"""
        
        company_type = company_type.lower()
        if company_type not in ["startup", "midsize", "enterprise"]:
            company_type = "midsize"
        
        # If has connection, always recommend connection-focused
        if has_connection:
            success_rate = self.style_performance["connection_focused"][company_type]
            return {
                "recommended_style": "connection_focused",
                "success_rate": success_rate,
                "confidence": success_rate * 100
            }
        
        # Otherwise, find best style for this company type
        best_style = max(
            self.style_performance.items(),
            key=lambda x: x[1][company_type]
        )
        
        return {
            "recommended_style": best_style[0],
            "success_rate": best_style[1][company_type],
            "confidence": best_style[1][company_type] * 100
        }
    
    def coach(self, draft_message, company_type, has_connection=False, position=""):
        """
        Analyze and improve a follow-up message
        
        Args:
            draft_message: The user's draft follow-up message
            company_type: "startup", "midsize", or "enterprise"
            has_connection: Boolean, do you have a mutual connection?
            position: Job title (optional, for context)
        
        Returns:
            Dict with score, feedback, and improved version
        """
        
        print(f"\n{'='*70}")
        print(f"MESSAGE COACH CHAIN")
        print(f"{'='*70}")
        print(f"Company Type: {company_type}")
        print(f"Has Connection: {has_connection}")
        print(f"Position: {position if position else 'Not specified'}")
        print(f"\nDRAFT MESSAGE:")
        print(f"{draft_message}")
        print(f"{'='*70}\n")
        
        # Step 1: Get RL style recommendation
        rl_rec = self.get_rl_style_recommendation(company_type, has_connection)
        print(f"📊 RL Recommendation: Use '{rl_rec['recommended_style']}' style (Success rate: {rl_rec['success_rate']:.1%})")
        
        # Step 2: Query RAG for message best practices
        rag_query = f"What makes a good follow-up email for a {company_type} company? How should I structure it?"
        
        print(f"🔍 Querying RAG: '{rag_query}'")
        rag_result = self.rag.query(rag_query, k=3)
        print(f"✅ RAG retrieved {len(rag_result['sources'])} sources")
        
        # Step 3: Analyze and improve with LLM
        prompt = self._build_coaching_prompt(
            draft_message, company_type, has_connection, position,
            rl_rec, rag_result
        )
        
        print(f"🤖 Analyzing message with GPT-4o-mini...")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 1500,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            analysis = data['choices'][0]['message']['content']
        else:
            analysis = f"Error analyzing message: {response.status_code}"
        
        print(f"✅ Analysis complete!\n")
        
        # Parse the analysis (expecting JSON format)
        try:
            result = json.loads(analysis)
        except:
            # If not JSON, return raw analysis
            result = {
                "score": 5,
                "feedback": analysis,
                "improved_message": "Could not generate improved version"
            }
        
        return {
            "original_message": draft_message,
            "score": result.get("score", 5),
            "feedback": result.get("feedback", []),
            "improved_message": result.get("improved_message", ""),
            "recommended_style": rl_rec['recommended_style'],
            "rl_confidence": rl_rec['confidence'],
            "rag_sources": rag_result['sources']
        }
    
    def _build_coaching_prompt(self, draft, company_type, has_connection, position, rl_rec, rag_result):
        """Build prompt for message coaching"""
        
        connection_str = "with a mutual connection" if has_connection else "cold (no connection)"
        
        prompt = f"""You are an expert career coach analyzing a follow-up email for a {company_type} job application ({connection_str}).

DRAFT MESSAGE:
{draft}

CONTEXT:
- Company type: {company_type}
- Connection status: {"Has mutual connection" if has_connection else "No connection"}
- Position: {position if position else "Not specified"}

RL SYSTEM RECOMMENDATION:
- Recommended style: {rl_rec['recommended_style']}
- Success rate: {rl_rec['success_rate']:.1%}
- This style has proven most effective based on 500 training episodes

BEST PRACTICES FROM KNOWLEDGE BASE:
{rag_result['answer']}

YOUR TASK:
Analyze the draft message and provide a detailed coaching response in JSON format:

{{
  "score": <1-10 integer score>,
  "feedback": [
    "Specific issue 1",
    "Specific issue 2",
    "Specific issue 3",
    "What's working well"
  ],
  "improved_message": "<rewritten version incorporating all improvements>",
  "style_alignment": "<how well it matches the recommended style>"
}}

SCORING CRITERIA:
- 1-3: Poor (major issues, unprofessional)
- 4-5: Below average (several improvements needed)
- 6-7: Good (solid but could be better)
- 8-9: Excellent (minor tweaks only)
- 10: Perfect (no improvements needed)

Focus on:
1. Length (50-150 words optimal)
2. Subject line quality
3. Opening strength
4. Value proposition
5. Call to action clarity
6. Professional tone
7. Style alignment with RL recommendation

Return ONLY valid JSON, no other text.
"""
        
        return prompt


def test_message_coach():
    """Test the message coach chain"""
    
    coach = MessageCoachChain()
    
    # Test case 1: Poor startup message
    print("\n" + "="*70)
    print("TEST 1: Poor startup message")
    print("="*70)
    
    poor_message = """Hi,
    
I applied to your company last week and wanted to follow up. I'm really interested in the position and think I'd be a great fit. Can you let me know the status of my application?

Thanks,
John"""
    
    result1 = coach.coach(
        draft_message=poor_message,
        company_type="startup",
        has_connection=False,
        position="Data Analyst"
    )
    
    print(f"\nSCORE: {result1['score']}/10")
    print(f"RECOMMENDED STYLE: {result1['recommended_style']}")
    print(f"\nFEEDBACK:")
    for fb in result1.get('feedback', []):
        print(f"  • {fb}")
    print(f"\nIMPROVED MESSAGE:\n{result1['improved_message']}")
    
    # Test case 2: Better enterprise message with connection
    print("\n" + "="*70)
    print("TEST 2: Good enterprise message with connection")
    print("="*70)
    
    good_message = """Subject: Following Up on Data Analyst Application – Referred by Sarah Chen

Hi Ms. Johnson,

Sarah Chen from your Analytics team suggested I reach out regarding the Senior Data Analyst position. I applied last week and wanted to express my continued interest.

My experience in building predictive models that increased customer retention by 23% aligns closely with the role's requirements. I'd welcome the opportunity to discuss how I could contribute to your team's success.

Would you be available for a brief conversation this week?

Best regards,
Jane Smith"""
    
    result2 = coach.coach(
        draft_message=good_message,
        company_type="enterprise",
        has_connection=True,
        position="Senior Data Analyst"
    )
    
    print(f"\nSCORE: {result2['score']}/10")
    print(f"RECOMMENDED STYLE: {result2['recommended_style']}")
    print(f"\nFEEDBACK:")
    for fb in result2.get('feedback', []):
        print(f"  • {fb}")
    print(f"\nIMPROVED MESSAGE:\n{result2['improved_message']}")
    
    # Save results
    results = {
        "test1_poor_startup": {
            "original": result1['original_message'],
            "score": result1['score'],
            "feedback": result1.get('feedback', []),
            "improved": result1['improved_message'],
            "recommended_style": result1['recommended_style']
        },
        "test2_good_enterprise": {
            "original": result2['original_message'],
            "score": result2['score'],
            "feedback": result2.get('feedback', []),
            "improved": result2['improved_message'],
            "recommended_style": result2['recommended_style']
        }
    }
    
    with open("results/message_coach_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to results/message_coach_tests.json")
    print("="*70)


if __name__ == "__main__":
    test_message_coach()
