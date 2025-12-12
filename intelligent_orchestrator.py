"""
Intelligent Orchestrator
Routes queries to appropriate chains and synthesizes results
"""

import json
import os
import sys
from prompt_chains.timing_advisor_chain import TimingAdvisorChain
from prompt_chains.message_coach_chain import MessageCoachChain
from prompt_chains.strategy_synthesizer_chain import StrategySynthesizerChain
from prompt_chains.career_qa_chain import CareerQAChain
from prompt_chains.confidence_explainer_chain import ConfidenceExplainerChain


class IntelligentOrchestrator:
    """
    Master orchestrator that:
    1. Analyzes user queries
    2. Routes to appropriate chain(s)
    3. Combines results intelligently
    4. Provides unified responses
    """
    
    def __init__(self):
        # Initialize all chains
        self.timing_advisor = TimingAdvisorChain()
        self.message_coach = MessageCoachChain()
        self.strategy_synthesizer = StrategySynthesizerChain()
        self.career_qa = CareerQAChain()
        self.confidence_explainer = ConfidenceExplainerChain()
        
        print("✅ Intelligent Orchestrator initialized with 5 chains")
    
    def process(self, query_type, query_data):
        """
        Process a query by routing to appropriate chain(s)
        
        Args:
            query_type: str, one of:
                - "timing_advice" - When to follow up
                - "message_review" - Review/improve a message
                - "full_strategy" - Complete application strategy
                - "career_question" - General career Q&A
                - "explain_recommendation" - Explain RL metrics
            
            query_data: dict with relevant data for the query
        
        Returns:
            Unified response dict
        """
        
        print(f"\n{'='*70}")
        print(f"INTELLIGENT ORCHESTRATOR")
        print(f"{'='*70}")
        print(f"Query Type: {query_type}")
        print(f"{'='*70}\n")
        
        if query_type == "timing_advice":
            return self._handle_timing_advice(query_data)
        
        elif query_type == "message_review":
            return self._handle_message_review(query_data)
        
        elif query_type == "full_strategy":
            return self._handle_full_strategy(query_data)
        
        elif query_type == "career_question":
            return self._handle_career_question(query_data)
        
        elif query_type == "explain_recommendation":
            return self._handle_explain_recommendation(query_data)
        
        else:
            return {
                "error": f"Unknown query type: {query_type}",
                "supported_types": [
                    "timing_advice",
                    "message_review",
                    "full_strategy",
                    "career_question",
                    "explain_recommendation"
                ]
            }
    
    def _handle_timing_advice(self, data):
        """Route to Timing Advisor chain"""
        print("🎯 Routing to: TIMING ADVISOR")
        
        result = self.timing_advisor.advise(
            company_type=data.get("company_type", "midsize"),
            has_connection=data.get("has_connection", False),
            current_day=data.get("current_day", 0)
        )
        
        # Also explain the recommendation
        print("\n📖 Adding explanation...")
        explanation = self.confidence_explainer.explain(
            "timing",
            {
                "wait_time": result['recommendation'],
                "q_value": result['q_value'],
                "confidence": result['confidence'],
                "company_type": data.get("company_type", "midsize")
            }
        )
        
        return {
            "query_type": "timing_advice",
            "recommendation": result['recommendation'],
            "confidence": result['confidence'],
            "reasoning": result['reasoning'],
            "should_act_now": result['should_act_now'],
            "plain_explanation": explanation['explanation'],
            "chain_used": "Timing Advisor + Confidence Explainer"
        }
    
    def _handle_message_review(self, data):
        """Route to Message Coach chain"""
        print("🎯 Routing to: MESSAGE COACH")
        
        result = self.message_coach.coach(
            draft_message=data.get("message", ""),
            company_type=data.get("company_type", "midsize"),
            has_connection=data.get("has_connection", False),
            position=data.get("position", "")
        )
        
        # Also explain the style recommendation
        print("\n📖 Adding style explanation...")
        style_explanation = self.confidence_explainer.explain(
            "style",
            {
                "style": result['recommended_style'],
                "confidence": result['rl_confidence'],
                "company_type": data.get("company_type", "midsize")
            }
        )
        
        return {
            "query_type": "message_review",
            "original_score": result['score'],
            "feedback": result['feedback'],
            "improved_message": result['improved_message'],
            "recommended_style": result['recommended_style'],
            "style_explanation": style_explanation['explanation'],
            "chain_used": "Message Coach + Confidence Explainer"
        }
    
    def _handle_full_strategy(self, data):
        """Route to Strategy Synthesizer (master chain)"""
        print("🎯 Routing to: STRATEGY SYNTHESIZER (Master Chain)")
        
        result = self.strategy_synthesizer.synthesize(data)
        
        return {
            "query_type": "full_strategy",
            "company": result['company_name'],
            "position": result['position'],
            "comprehensive_strategy": result['comprehensive_strategy'],
            "timing_recommendation": result['rl_timing']['wait_time'],
            "style_recommendation": result['rl_style']['style'],
            "should_act_now": result['should_act_now'],
            "sources_consulted": result['sources_consulted'],
            "chain_used": "Strategy Synthesizer (RL + RAG + Multi-step)"
        }
    
    def _handle_career_question(self, data):
        """Route to Career Q&A chain"""
        print("🎯 Routing to: CAREER Q&A")
        
        result = self.career_qa.ask(data.get("question", ""))
        
        return {
            "query_type": "career_question",
            "question": result['question'],
            "answer": result['answer'],
            "sources": [s['filename'] for s in result['sources']],
            "chain_used": "Career Q&A (Pure RAG)"
        }
    
    def _handle_explain_recommendation(self, data):
        """Route to Confidence Explainer"""
        print("🎯 Routing to: CONFIDENCE EXPLAINER")
        
        result = self.confidence_explainer.explain(
            recommendation_type=data.get("type", "timing"),
            metric_data=data.get("metrics", {})
        )
        
        return {
            "query_type": "explain_recommendation",
            "recommendation_type": result['recommendation_type'],
            "explanation": result['explanation'],
            "chain_used": "Confidence Explainer"
        }


def test_orchestrator():
    """Test the intelligent orchestrator with various queries"""
    
    orchestrator = IntelligentOrchestrator()
    
    test_cases = [
        {
            "name": "Timing Advice for Startup",
            "query_type": "timing_advice",
            "data": {
                "company_type": "startup",
                "has_connection": True,
                "current_day": 0
            }
        },
        {
            "name": "Message Review",
            "query_type": "message_review",
            "data": {
                "message": "Hi, I wanted to follow up on my application. Thanks!",
                "company_type": "enterprise",
                "has_connection": False,
                "position": "Data Analyst"
            }
        },
        {
            "name": "Full Strategy",
            "query_type": "full_strategy",
            "data": {
                "company_name": "TechCorp",
                "company_type": "midsize",
                "position": "Senior Data Analyst",
                "has_connection": False,
                "days_since_application": 3
            }
        },
        {
            "name": "Career Question",
            "query_type": "career_question",
            "data": {
                "question": "What are the most important SQL concepts for interviews?"
            }
        }
    ]
    
    results = {}
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}/{len(test_cases)}: {test['name']}")
        print(f"{'='*70}")
        
        result = orchestrator.process(test['query_type'], test['data'])
        
        print(f"\n✅ RESULT:")
        print(f"Chain Used: {result.get('chain_used', 'N/A')}")
        
        if test['query_type'] == 'timing_advice':
            print(f"Recommendation: {result['recommendation']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"Should Act Now: {result['should_act_now']}")
            print(f"\nExplanation: {result['plain_explanation'][:200]}...")
        
        elif test['query_type'] == 'message_review':
            print(f"Score: {result['original_score']}/10")
            print(f"Recommended Style: {result['recommended_style']}")
            print(f"Feedback: {len(result.get('feedback', []))} points")
        
        elif test['query_type'] == 'full_strategy':
            print(f"Company: {result['company']}")
            print(f"Position: {result['position']}")
            print(f"Timing: {result['timing_recommendation']}")
            print(f"Style: {result['style_recommendation']}")
        
        elif test['query_type'] == 'career_question':
            print(f"Question: {result['question']}")
            print(f"Sources: {len(result['sources'])} documents")
        
        results[f"test_{i}_{test['name'].lower().replace(' ', '_')}"] = {
            "query_type": result.get('query_type'),
            "chain_used": result.get('chain_used'),
            "summary": f"Processed successfully with {result.get('chain_used', 'unknown chain')}"
        }
    
    # Save results
    with open("results/orchestrator_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ ALL TESTS COMPLETE!")
    print(f"✅ Results saved to results/orchestrator_tests.json")
    print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print(f"ORCHESTRATOR CAPABILITIES:")
    print(f"{'='*70}")
    print(f"✅ Routes queries to 5 specialized chains")
    print(f"✅ Combines multiple chains when needed")
    print(f"✅ Provides unified, consistent responses")
    print(f"✅ Explains recommendations in plain English")
    print(f"✅ Handles 5 different query types")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_orchestrator()
