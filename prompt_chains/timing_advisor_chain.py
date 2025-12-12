"""
Timing Advisor Chain
Synthesizes RL timing recommendations with RAG knowledge base guidance
"""

import json
import requests
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advanced_rag_system import AdvancedRAGSystem


class TimingAdvisorChain:
    """
    Intelligent timing recommendation chain that combines:
    - RL Q-Learning optimal wait times
    - RAG knowledge base timing strategies
    - Company-specific context
    """
    
    def __init__(self):
        self.rag = AdvancedRAGSystem()
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # RL Q-values (from your take-home project)
        self.q_values = {
            "startup": {
                "1-3 days": 10.83,
                "3-5 days": 8.42,
                "5-7 days": 6.15,
                "7-10 days": 3.91
            },
            "midsize": {
                "1-3 days": 7.25,
                "3-5 days": 9.18,
                "5-7 days": 8.67,
                "7-10 days": 6.42
            },
            "enterprise": {
                "1-3 days": 3.12,
                "3-5 days": 5.67,
                "5-7 days": 7.89,
                "7-10 days": 4.06
            }
        }
    
    def get_rl_recommendation(self, company_type, has_connection):
        """Get RL-based timing recommendation"""
        
        company_type = company_type.lower()
        if company_type not in self.q_values:
            company_type = "midsize"  # default
        
        # Get Q-values for this company type
        q_vals = self.q_values[company_type]
        
        # Find optimal timing
        optimal_timing = max(q_vals.items(), key=lambda x: x[1])
        wait_time = optimal_timing[0]
        q_value = optimal_timing[1]
        
        # Calculate confidence (normalize Q-value to 0-100%)
        max_q = max(q_vals.values())
        min_q = min(q_vals.values())
        confidence = ((q_value - min_q) / (max_q - min_q)) * 100 if max_q > min_q else 85.0
        
        # Adjust for connection
        if has_connection and company_type == "startup":
            wait_time = "1-3 days"
            confidence = min(confidence + 10, 95)
        
        return {
            "wait_time": wait_time,
            "q_value": q_value,
            "confidence": confidence,
            "company_type": company_type
        }
    
    def advise(self, company_type, has_connection=False, current_day=0):
        """
        Generate intelligent timing advice
        
        Args:
            company_type: "startup", "midsize", or "enterprise"
            has_connection: Boolean, do you have a mutual connection?
            current_day: Days since application (0 = just applied)
        
        Returns:
            Dict with recommendation, reasoning, and sources
        """
        
        print(f"\n{'='*70}")
        print(f"TIMING ADVISOR CHAIN")
        print(f"{'='*70}")
        print(f"Company Type: {company_type}")
        print(f"Has Connection: {has_connection}")
        print(f"Days Since Application: {current_day}")
        print(f"{'='*70}\n")
        
        # Step 1: Get RL recommendation
        rl_rec = self.get_rl_recommendation(company_type, has_connection)
        print(f"📊 RL Recommendation: {rl_rec['wait_time']} (Q-value: {rl_rec['q_value']:.2f}, Confidence: {rl_rec['confidence']:.1f}%)")
        
        # Step 2: Query RAG for timing knowledge
        connection_status = "with a connection" if has_connection else "cold (no connection)"
        rag_query = f"When should I follow up with a {company_type} company {connection_status}?"
        
        print(f"🔍 Querying RAG: '{rag_query}'")
        rag_result = self.rag.query(rag_query, k=3)
        print(f"✅ RAG retrieved {len(rag_result['sources'])} sources")
        
        # Step 3: Synthesize with LLM
        prompt = self._build_synthesis_prompt(
            company_type, has_connection, current_day,
            rl_rec, rag_result
        )
        
        print(f"🤖 Generating synthesis with GPT-4o-mini...")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 1000,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            synthesis = data['choices'][0]['message']['content']
        else:
            synthesis = f"Error generating synthesis: {response.status_code}"
        
        print(f"✅ Synthesis complete!\n")
        
        return {
            "recommendation": rl_rec['wait_time'],
            "confidence": rl_rec['confidence'],
            "q_value": rl_rec['q_value'],
            "reasoning": synthesis,
            "rl_data": rl_rec,
            "rag_sources": rag_result['sources'],
            "should_act_now": current_day >= int(rl_rec['wait_time'].split('-')[0])
        }
    
    def _build_synthesis_prompt(self, company_type, has_connection, current_day, rl_rec, rag_result):
        """Build prompt for LLM synthesis"""
        
        connection_str = "with a mutual connection" if has_connection else "as a cold application (no connection)"
        
        prompt = f"""You are an expert career advisor helping a job seeker decide when to follow up on a {company_type} job application ({connection_str}).

CURRENT SITUATION:
- Applied {current_day} days ago
- Company type: {company_type}
- Connection status: {"Has mutual connection" if has_connection else "No connection (cold)"}

RL SYSTEM RECOMMENDATION:
- Optimal timing: {rl_rec['wait_time']}
- Confidence: {rl_rec['confidence']:.1f}%
- Q-value: {rl_rec['q_value']:.2f}
- This is based on 500 training episodes analyzing response patterns

KNOWLEDGE BASE GUIDANCE:
{rag_result['answer']}

YOUR TASK:
Synthesize the RL recommendation and knowledge base guidance into a clear, actionable recommendation. Include:

1. **When to follow up** (specific day recommendation)
2. **Why this timing works** (explain the reasoning)
3. **What to do if you've already waited too long** (recovery strategy)
4. **Key tips for this specific situation**

Be concise (200-250 words), confident, and practical. Use "you should" language.
"""
        
        return prompt


def test_timing_advisor():
    """Test the timing advisor chain"""
    
    advisor = TimingAdvisorChain()
    
    # Test case 1: Startup with connection
    print("\n" + "="*70)
    print("TEST 1: Startup with connection, day 0")
    print("="*70)
    
    result1 = advisor.advise(
        company_type="startup",
        has_connection=True,
        current_day=0
    )
    
    print(f"\nRECOMMENDATION: {result1['recommendation']}")
    print(f"CONFIDENCE: {result1['confidence']:.1f}%")
    print(f"SHOULD ACT NOW: {result1['should_act_now']}")
    print(f"\nREASONING:\n{result1['reasoning']}")
    
    # Test case 2: Enterprise cold application
    print("\n" + "="*70)
    print("TEST 2: Enterprise cold, day 5")
    print("="*70)
    
    result2 = advisor.advise(
        company_type="enterprise",
        has_connection=False,
        current_day=5
    )
    
    print(f"\nRECOMMENDATION: {result2['recommendation']}")
    print(f"CONFIDENCE: {result2['confidence']:.1f}%")
    print(f"SHOULD ACT NOW: {result2['should_act_now']}")
    print(f"\nREASONING:\n{result2['reasoning']}")
    
    # Save results
    results = {
        "test1_startup_connection": {
            "recommendation": result1['recommendation'],
            "confidence": result1['confidence'],
            "q_value": result1['q_value'],
            "reasoning": result1['reasoning'],
            "should_act_now": result1['should_act_now']
        },
        "test2_enterprise_cold": {
            "recommendation": result2['recommendation'],
            "confidence": result2['confidence'],
            "q_value": result2['q_value'],
            "reasoning": result2['reasoning'],
            "should_act_now": result2['should_act_now']
        }
    }
    
    with open("results/timing_advisor_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to results/timing_advisor_tests.json")
    print("="*70)


if __name__ == "__main__":
    test_timing_advisor()
