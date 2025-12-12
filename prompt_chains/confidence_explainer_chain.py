"""
Confidence Explainer Chain
Translates RL technical metrics into user-friendly explanations
"""

import json
import requests
import os


class ConfidenceExplainerChain:
    """
    Makes RL recommendations understandable to non-technical users
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
    
    def explain(self, recommendation_type, metric_data):
        """
        Explain why the RL system made a specific recommendation
        
        Args:
            recommendation_type: "timing" or "style"
            metric_data: Dict with RL metrics (q_value, confidence, success_rate, etc.)
        
        Returns:
            Plain English explanation
        """
        
        print(f"\n{'='*70}")
        print(f"CONFIDENCE EXPLAINER CHAIN")
        print(f"{'='*70}")
        print(f"Type: {recommendation_type}")
        print(f"Metrics: {metric_data}")
        print(f"{'='*70}\n")
        
        prompt = self._build_explanation_prompt(recommendation_type, metric_data)
        
        print(f"🤖 Generating explanation with GPT-4o-mini...")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 500,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            explanation = data['choices'][0]['message']['content']
        else:
            explanation = f"Error generating explanation: {response.status_code}"
        
        print(f"✅ Explanation complete!\n")
        print(f"EXPLANATION:\n{explanation}")
        
        return {
            "recommendation_type": recommendation_type,
            "metrics": metric_data,
            "explanation": explanation
        }
    
    def _build_explanation_prompt(self, rec_type, metrics):
        """Build explanation prompt"""
        
        if rec_type == "timing":
            prompt = f"""You are explaining a job application follow-up timing recommendation to a non-technical user.

TECHNICAL METRICS:
- Recommended timing: {metrics.get('wait_time', 'N/A')}
- Q-value: {metrics.get('q_value', 0):.2f}
- Confidence: {metrics.get('confidence', 0):.1f}%
- Company type: {metrics.get('company_type', 'N/A')}

Q-VALUE CONTEXT:
- Q-values represent expected success of different actions
- Higher Q-value = better expected outcome
- Learned from 500 simulated job applications
- Range typically 3-11 (higher is better)

YOUR TASK:
Explain in 2-3 sentences why this timing recommendation makes sense, WITHOUT using technical jargon like "Q-value" or "reinforcement learning". 

Focus on:
- Why this timing works well
- What patterns were discovered
- How confident we are

Use everyday language like "our analysis of 500 applications found that..." or "this timing has proven most effective because..."
"""
        
        else:  # style
            prompt = f"""You are explaining a message style recommendation to a non-technical user.

TECHNICAL METRICS:
- Recommended style: {metrics.get('style', 'N/A')}
- Success rate: {metrics.get('success_rate', 0):.1%}
- Confidence: {metrics.get('confidence', 0):.1f}%
- Company type: {metrics.get('company_type', 'N/A')}

SUCCESS RATE CONTEXT:
- Success rate = % of applications that got responses
- Based on Thompson Sampling algorithm
- Learned from 500 training episodes
- Continuously adapts based on results

YOUR TASK:
Explain in 2-3 sentences why this message style is recommended, WITHOUT using technical terms like "Thompson Sampling" or "success rate optimization".

Focus on:
- Why this style works for this company type
- What makes it effective
- How confident we are

Use everyday language like "based on analyzing hundreds of applications, we found that..." or "this style tends to get the best response because..."
"""
        
        return prompt


def test_confidence_explainer():
    """Test the confidence explainer chain"""
    
    explainer = ConfidenceExplainerChain()
    
    # Test case 1: Timing explanation
    print("\n" + "="*70)
    print("TEST 1: Explain timing recommendation")
    print("="*70)
    
    timing_metrics = {
        "wait_time": "1-3 days",
        "q_value": 10.83,
        "confidence": 95.0,
        "company_type": "startup"
    }
    
    result1 = explainer.explain("timing", timing_metrics)
    
    # Test case 2: Style explanation
    print("\n" + "="*70)
    print("TEST 2: Explain style recommendation")
    print("="*70)
    
    style_metrics = {
        "style": "casual",
        "success_rate": 0.733,
        "confidence": 73.3,
        "company_type": "startup"
    }
    
    result2 = explainer.explain("style", style_metrics)
    
    # Test case 3: Enterprise formal style
    print("\n" + "="*70)
    print("TEST 3: Explain enterprise style")
    print("="*70)
    
    style_metrics2 = {
        "style": "formal",
        "success_rate": 0.417,
        "confidence": 41.7,
        "company_type": "enterprise"
    }
    
    result3 = explainer.explain("style", style_metrics2)
    
    # Save results
    results = {
        "test1_timing": {
            "metrics": result1['metrics'],
            "explanation": result1['explanation']
        },
        "test2_casual_style": {
            "metrics": result2['metrics'],
            "explanation": result2['explanation']
        },
        "test3_formal_style": {
            "metrics": result3['metrics'],
            "explanation": result3['explanation']
        }
    }
    
    with open("results/confidence_explainer_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to results/confidence_explainer_tests.json")
    print("="*70)


if __name__ == "__main__":
    test_confidence_explainer()
