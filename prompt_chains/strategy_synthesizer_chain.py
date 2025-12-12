"""
Strategy Synthesizer Chain
The master chain that combines RL + RAG + all insights into comprehensive action plans
"""

import json
import requests
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advanced_rag_system import AdvancedRAGSystem


class StrategySynthesizerChain:
    """
    Master intelligence chain that synthesizes:
    - RL timing recommendations (Q-Learning)
    - RL style recommendations (Thompson Sampling)
    - RAG career guidance
    - Multi-step action planning
    """
    
    def __init__(self):
        self.rag = AdvancedRAGSystem()
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # RL Q-values for timing
        self.q_values = {
            "startup": {"1-3 days": 10.83, "3-5 days": 8.42, "5-7 days": 6.15, "7-10 days": 3.91},
            "midsize": {"1-3 days": 7.25, "3-5 days": 9.18, "5-7 days": 8.67, "7-10 days": 6.42},
            "enterprise": {"1-3 days": 3.12, "3-5 days": 5.67, "5-7 days": 7.89, "7-10 days": 4.06}
        }
        
        # RL Thompson Sampling success rates
        self.style_performance = {
            "formal": {"enterprise": 0.417, "midsize": 0.358, "startup": 0.283},
            "casual": {"enterprise": 0.267, "midsize": 0.408, "startup": 0.733},
            "connection_focused": {"enterprise": 0.550, "midsize": 0.625, "startup": 0.700}
        }
    
    def get_comprehensive_recommendations(self, company_type, has_connection):
        """Get both timing and style recommendations from RL"""
        
        company_type = company_type.lower()
        if company_type not in self.q_values:
            company_type = "midsize"
        
        # Timing recommendation
        optimal_timing = max(self.q_values[company_type].items(), key=lambda x: x[1])
        timing_rec = {
            "wait_time": optimal_timing[0],
            "q_value": optimal_timing[1],
            "confidence": (optimal_timing[1] / max(self.q_values[company_type].values())) * 100
        }
        
        # Style recommendation
        if has_connection:
            style_rec = {
                "style": "connection_focused",
                "success_rate": self.style_performance["connection_focused"][company_type],
                "confidence": self.style_performance["connection_focused"][company_type] * 100
            }
        else:
            best_style = max(self.style_performance.items(), key=lambda x: x[1][company_type])
            style_rec = {
                "style": best_style[0],
                "success_rate": best_style[1][company_type],
                "confidence": best_style[1][company_type] * 100
            }
        
        return timing_rec, style_rec
    
    def synthesize(self, job_details):
        """
        Create comprehensive application strategy
        
        Args:
            job_details: Dict with keys:
                - company_name: str
                - company_type: "startup", "midsize", or "enterprise"
                - position: str
                - has_connection: bool
                - connection_name: str (optional)
                - days_since_application: int
                - current_situation: str (optional context)
        
        Returns:
            Complete action plan with timing, messaging, and research strategies
        """
        
        company_name = job_details.get("company_name", "the company")
        company_type = job_details.get("company_type", "midsize")
        position = job_details.get("position", "the position")
        has_connection = job_details.get("has_connection", False)
        connection_name = job_details.get("connection_name", "")
        days_since = job_details.get("days_since_application", 0)
        situation = job_details.get("current_situation", "")
        
        print(f"\n{'='*70}")
        print(f"STRATEGY SYNTHESIZER CHAIN")
        print(f"{'='*70}")
        print(f"Company: {company_name} ({company_type})")
        print(f"Position: {position}")
        print(f"Connection: {connection_name if has_connection else 'None'}")
        print(f"Days Since Application: {days_since}")
        print(f"{'='*70}\n")
        
        # Step 1: Get RL recommendations
        timing_rec, style_rec = self.get_comprehensive_recommendations(company_type, has_connection)
        print(f"📊 RL Timing: {timing_rec['wait_time']} (Q={timing_rec['q_value']:.2f}, Confidence={timing_rec['confidence']:.1f}%)")
        print(f"📊 RL Style: {style_rec['style']} (Success={style_rec['success_rate']:.1%}, Confidence={style_rec['confidence']:.1f}%)")
        
        # Step 2: Query RAG for multiple aspects
        print(f"\n🔍 Querying RAG for comprehensive guidance...")
        
        # Query 1: Timing strategy
        timing_query = f"When and how should I follow up with a {company_type} company?"
        timing_rag = self.rag.query(timing_query, k=2)
        
        # Query 2: Message strategy
        message_query = f"How should I write a follow-up message for a {company_type} company in {style_rec['style']} style?"
        message_rag = self.rag.query(message_query, k=2)
        
        # Query 3: Research strategy
        research_query = f"How should I research a {company_type} company before following up?"
        research_rag = self.rag.query(research_query, k=2)
        
        print(f"✅ Retrieved guidance on: Timing, Messaging, Research")
        
        # Step 3: Synthesize everything into action plan
        prompt = self._build_synthesis_prompt(
            job_details, timing_rec, style_rec,
            timing_rag, message_rag, research_rag
        )
        
        print(f"\n🤖 Generating comprehensive strategy with GPT-4o-mini...")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 2000,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            strategy = data['choices'][0]['message']['content']
        else:
            strategy = f"Error generating strategy: {response.status_code}"
        
        print(f"✅ Strategy complete!\n")
        
        return {
            "company_name": company_name,
            "position": position,
            "rl_timing": timing_rec,
            "rl_style": style_rec,
            "comprehensive_strategy": strategy,
            "should_act_now": days_since >= int(timing_rec['wait_time'].split('-')[0]),
            "sources_consulted": {
                "timing": [s['filename'] for s in timing_rag['sources']],
                "messaging": [s['filename'] for s in message_rag['sources']],
                "research": [s['filename'] for s in research_rag['sources']]
            }
        }
    
    def _build_synthesis_prompt(self, job_details, timing_rec, style_rec, timing_rag, message_rag, research_rag):
        """Build master synthesis prompt"""
        
        company_name = job_details.get("company_name", "the company")
        company_type = job_details.get("company_type", "midsize")
        position = job_details.get("position", "the position")
        has_connection = job_details.get("has_connection", False)
        connection_name = job_details.get("connection_name", "")
        days_since = job_details.get("days_since_application", 0)
        
        prompt = f"""You are an expert career strategist creating a comprehensive action plan for a job application.

APPLICATION DETAILS:
- Company: {company_name} ({company_type})
- Position: {position}
- Connection: {connection_name if has_connection else "None (cold application)"}
- Days since application: {days_since}

RL INTELLIGENCE (from 500 training episodes):

TIMING RECOMMENDATION:
- Optimal timing: {timing_rec['wait_time']}
- Q-value: {timing_rec['q_value']:.2f}
- Confidence: {timing_rec['confidence']:.1f}%

STYLE RECOMMENDATION:
- Optimal style: {style_rec['style']}
- Success rate: {style_rec['success_rate']:.1%}
- Confidence: {style_rec['confidence']:.1f}%

KNOWLEDGE BASE GUIDANCE:

TIMING STRATEGY:
{timing_rag['answer'][:500]}

MESSAGING STRATEGY:
{message_rag['answer'][:500]}

RESEARCH STRATEGY:
{research_rag['answer'][:500]}

YOUR TASK:
Create a comprehensive, actionable strategy document with these sections:

## IMMEDIATE ACTION
What should they do RIGHT NOW? (considering they applied {days_since} days ago)

## TIMING STRATEGY
- When to send follow-up (specific day/date)
- Why this timing is optimal
- Backup plan if no response

## MESSAGE STRATEGY
- Message style to use ({style_rec['style']})
- Key points to include
- Subject line suggestion
- Template structure

## RESEARCH CHECKLIST
Before following up, research:
- [ ] Specific things to look up
- [ ] Red flags to watch for
- [ ] Ways to personalize the message

## NEXT STEPS TIMELINE
Day-by-day action plan for the next 2 weeks

## SUCCESS METRICS
How to know if the strategy is working

Keep it actionable, specific, and confident. Use markdown formatting.
Aim for 400-500 words total.
"""
        
        return prompt


def test_strategy_synthesizer():
    """Test the strategy synthesizer chain"""
    
    synthesizer = StrategySynthesizerChain()
    
    # Test case 1: Startup with connection, just applied
    print("\n" + "="*70)
    print("TEST 1: Startup with connection - just applied")
    print("="*70)
    
    job1 = {
        "company_name": "TechFlow AI",
        "company_type": "startup",
        "position": "Data Analyst",
        "has_connection": True,
        "connection_name": "Alex Chen",
        "days_since_application": 0,
        "current_situation": "Just submitted application through company website"
    }
    
    result1 = synthesizer.synthesize(job1)
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE STRATEGY FOR {result1['company_name']}")
    print(f"{'='*70}")
    print(f"Position: {result1['position']}")
    print(f"RL Timing: {result1['rl_timing']['wait_time']} (Confidence: {result1['rl_timing']['confidence']:.1f}%)")
    print(f"RL Style: {result1['rl_style']['style']} (Success Rate: {result1['rl_style']['success_rate']:.1%})")
    print(f"Should Act Now: {result1['should_act_now']}")
    print(f"\n{result1['comprehensive_strategy']}")
    
    # Test case 2: Enterprise cold application, 7 days later
    print("\n" + "="*70)
    print("TEST 2: Enterprise cold - 7 days later")
    print("="*70)
    
    job2 = {
        "company_name": "DataCorp Industries",
        "company_type": "enterprise",
        "position": "Senior Business Intelligence Analyst",
        "has_connection": False,
        "days_since_application": 7,
        "current_situation": "Applied through LinkedIn, haven't heard back"
    }
    
    result2 = synthesizer.synthesize(job2)
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE STRATEGY FOR {result2['company_name']}")
    print(f"{'='*70}")
    print(f"Position: {result2['position']}")
    print(f"RL Timing: {result2['rl_timing']['wait_time']} (Confidence: {result2['rl_timing']['confidence']:.1f}%)")
    print(f"RL Style: {result2['rl_style']['style']} (Success Rate: {result2['rl_style']['success_rate']:.1%})")
    print(f"Should Act Now: {result2['should_act_now']}")
    print(f"\n{result2['comprehensive_strategy']}")
    
    # Save results
    results = {
        "test1_startup_connection": {
            "company": result1['company_name'],
            "timing": result1['rl_timing']['wait_time'],
            "style": result1['rl_style']['style'],
            "should_act_now": result1['should_act_now'],
            "strategy": result1['comprehensive_strategy']
        },
        "test2_enterprise_cold": {
            "company": result2['company_name'],
            "timing": result2['rl_timing']['wait_time'],
            "style": result2['rl_style']['style'],
            "should_act_now": result2['should_act_now'],
            "strategy": result2['comprehensive_strategy']
        }
    }
    
    with open("results/strategy_synthesizer_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to results/strategy_synthesizer_tests.json")
    print("="*70)


if __name__ == "__main__":
    test_strategy_synthesizer()
