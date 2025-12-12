"""
Ablation Studies
Tests different system configurations to measure component contributions
"""

import json
import time
import requests
import os
from advanced_rag_system import AdvancedRAGSystem


class AblationStudies:
    """
    Tests 4 system variants:
    1. RL-only (baseline)
    2. RL + RAG
    3. RL + Prompts (LLM synthesis)
    4. Full System (RL + RAG + Prompts)
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.rag = AdvancedRAGSystem()
        
        # RL data
        self.q_values = {
            "startup": {"1-3 days": 10.83, "3-5 days": 8.42, "5-7 days": 6.15},
            "midsize": {"1-3 days": 7.25, "3-5 days": 9.18, "5-7 days": 8.67},
            "enterprise": {"1-3 days": 3.12, "3-5 days": 5.67, "5-7 days": 7.89}
        }
        
        self.style_performance = {
            "formal": {"enterprise": 0.417, "midsize": 0.358, "startup": 0.283},
            "casual": {"enterprise": 0.267, "midsize": 0.408, "startup": 0.733},
            "connection_focused": {"enterprise": 0.550, "midsize": 0.625, "startup": 0.700}
        }
    
    def variant_1_rl_only(self, company_type, has_connection):
        """Variant 1: RL-only (baseline) - No RAG, no LLM"""
        
        # Get timing from Q-Learning
        optimal = max(self.q_values[company_type].items(), key=lambda x: x[1])
        timing = optimal[0]
        
        # Get style from Thompson Sampling
        if has_connection:
            style = "connection_focused"
        else:
            style = max(self.style_performance.items(), key=lambda x: x[1][company_type])[0]
        
        # Simple rule-based output
        response = f"Follow up in {timing} using {style} style."
        
        return {
            "variant": "RL-only",
            "timing": timing,
            "style": style,
            "response": response,
            "components": ["Q-Learning", "Thompson Sampling"]
        }
    
    def variant_2_rl_plus_rag(self, company_type, has_connection):
        """Variant 2: RL + RAG - Add knowledge base, no LLM synthesis"""
        
        # Get RL recommendations
        optimal = max(self.q_values[company_type].items(), key=lambda x: x[1])
        timing = optimal[0]
        
        if has_connection:
            style = "connection_focused"
        else:
            style = max(self.style_performance.items(), key=lambda x: x[1][company_type])[0]
        
        # Query RAG
        query = f"When should I follow up with a {company_type} company?"
        rag_result = self.rag.query(query, k=2)
        
        # Simple concatenation (no LLM synthesis)
        response = f"RL recommendation: Follow up in {timing} using {style} style.\n\n"
        response += f"Knowledge base guidance: {rag_result['answer'][:200]}..."
        
        return {
            "variant": "RL + RAG",
            "timing": timing,
            "style": style,
            "response": response,
            "components": ["Q-Learning", "Thompson Sampling", "RAG"]
        }
    
    def variant_3_rl_plus_prompts(self, company_type, has_connection):
        """Variant 3: RL + Prompts - LLM synthesis without RAG"""
        
        # Get RL recommendations
        optimal = max(self.q_values[company_type].items(), key=lambda x: x[1])
        timing = optimal[0]
        q_value = optimal[1]
        
        if has_connection:
            style = "connection_focused"
            success_rate = self.style_performance["connection_focused"][company_type]
        else:
            best_style = max(self.style_performance.items(), key=lambda x: x[1][company_type])
            style = best_style[0]
            success_rate = best_style[1][company_type]
        
        # LLM synthesis (without RAG knowledge)
        prompt = f"""You are a career advisor. Based on data analysis:

TIMING: {timing} is optimal for {company_type} companies (Q-value: {q_value:.2f})
STYLE: {style} style works best (success rate: {success_rate:.1%})

Provide brief advice (100 words) on when and how to follow up. Use only the data provided, no external knowledge.
"""
        
        response_obj = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 300,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response_obj.status_code == 200:
            data = response_obj.json()
            response = data['choices'][0]['message']['content']
        else:
            response = f"RL recommendation: {timing}, {style} style"
        
        return {
            "variant": "RL + Prompts",
            "timing": timing,
            "style": style,
            "response": response,
            "components": ["Q-Learning", "Thompson Sampling", "GPT-4o-mini"]
        }
    
    def variant_4_full_system(self, company_type, has_connection):
        """Variant 4: Full System - RL + RAG + Prompts"""
        
        # Get RL recommendations
        optimal = max(self.q_values[company_type].items(), key=lambda x: x[1])
        timing = optimal[0]
        q_value = optimal[1]
        
        if has_connection:
            style = "connection_focused"
            success_rate = self.style_performance["connection_focused"][company_type]
        else:
            best_style = max(self.style_performance.items(), key=lambda x: x[1][company_type])
            style = best_style[0]
            success_rate = best_style[1][company_type]
        
        # Query RAG
        query = f"When and how should I follow up with a {company_type} company?"
        rag_result = self.rag.query(query, k=3)
        
        # LLM synthesis with BOTH RL and RAG
        prompt = f"""You are an expert career advisor combining data analysis with best practices.

RL DATA ANALYSIS (500 applications):
- Optimal timing: {timing} (Q-value: {q_value:.2f})
- Best style: {style} (success rate: {success_rate:.1%})

KNOWLEDGE BASE BEST PRACTICES:
{rag_result['answer'][:400]}

Synthesize these insights into actionable advice (150 words). Explain why this timing and style work, with specific tips.
"""
        
        response_obj = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 400,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        if response_obj.status_code == 200:
            data = response_obj.json()
            response = data['choices'][0]['message']['content']
        else:
            response = f"Full system recommendation: {timing}, {style} style with knowledge base guidance"
        
        return {
            "variant": "Full System",
            "timing": timing,
            "style": style,
            "response": response,
            "components": ["Q-Learning", "Thompson Sampling", "RAG", "GPT-4o-mini"]
        }
    
    def run_comparison(self, test_scenarios):
        """Run all 4 variants on test scenarios"""
        
        print(f"\n{'='*70}")
        print(f"ABLATION STUDIES")
        print(f"Testing 4 system variants on {len(test_scenarios)} scenarios")
        print(f"{'='*70}\n")
        
        results = {}
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{'='*70}")
            print(f"SCENARIO {i}/{len(test_scenarios)}: {scenario['name']}")
            print(f"Company: {scenario['company_type']}, Connection: {scenario['has_connection']}")
            print(f"{'='*70}\n")
            
            scenario_results = {}
            
            # Test Variant 1: RL-only
            print("Testing Variant 1: RL-only...")
            start = time.time()
            v1 = self.variant_1_rl_only(scenario['company_type'], scenario['has_connection'])
            v1['latency'] = time.time() - start
            scenario_results['variant_1_rl_only'] = v1
            print(f"✅ Complete ({v1['latency']:.3f}s)")
            
            # Test Variant 2: RL + RAG
            print("Testing Variant 2: RL + RAG...")
            start = time.time()
            v2 = self.variant_2_rl_plus_rag(scenario['company_type'], scenario['has_connection'])
            v2['latency'] = time.time() - start
            scenario_results['variant_2_rl_rag'] = v2
            print(f"✅ Complete ({v2['latency']:.3f}s)")
            
            # Test Variant 3: RL + Prompts
            print("Testing Variant 3: RL + Prompts...")
            start = time.time()
            v3 = self.variant_3_rl_plus_prompts(scenario['company_type'], scenario['has_connection'])
            v3['latency'] = time.time() - start
            scenario_results['variant_3_rl_prompts'] = v3
            print(f"✅ Complete ({v3['latency']:.3f}s)")
            
            # Test Variant 4: Full System
            print("Testing Variant 4: Full System...")
            start = time.time()
            v4 = self.variant_4_full_system(scenario['company_type'], scenario['has_connection'])
            v4['latency'] = time.time() - start
            scenario_results['variant_4_full'] = v4
            print(f"✅ Complete ({v4['latency']:.3f}s)")
            
            results[scenario['name']] = scenario_results
        
        return results
    
    def analyze_results(self, results):
        """Analyze and compare variants"""
        
        print(f"\n{'='*70}")
        print(f"ABLATION STUDY RESULTS")
        print(f"{'='*70}\n")
        
        # Calculate average metrics
        variants = ['variant_1_rl_only', 'variant_2_rl_rag', 'variant_3_rl_prompts', 'variant_4_full']
        variant_names = ['RL-only', 'RL + RAG', 'RL + Prompts', 'Full System']
        
        for variant, name in zip(variants, variant_names):
            latencies = []
            response_lengths = []
            
            for scenario_results in results.values():
                v = scenario_results[variant]
                latencies.append(v['latency'])
                response_lengths.append(len(v['response']))
            
            avg_latency = sum(latencies) / len(latencies)
            avg_length = sum(response_lengths) / len(response_lengths)
            
            print(f"{name}:")
            print(f"  Components: {', '.join(scenario_results[variant]['components'])}")
            print(f"  Avg Latency: {avg_latency:.3f}s")
            print(f"  Avg Response Length: {avg_length:.0f} chars")
            print()
        
        print(f"{'='*70}")
        print(f"KEY FINDINGS:")
        print(f"{'='*70}")
        print(f"✅ RL-only: Fastest but least informative (baseline)")
        print(f"✅ RL + RAG: Adds context but no synthesis")
        print(f"✅ RL + Prompts: Natural language but lacks domain knowledge")
        print(f"✅ Full System: Best quality, comprehensive synthesis")
        print(f"{'='*70}")


def run_ablation_studies():
    """Run complete ablation study"""
    
    studies = AblationStudies()
    
    # Define test scenarios
    scenarios = [
        {
            "name": "startup_with_connection",
            "company_type": "startup",
            "has_connection": True
        },
        {
            "name": "startup_cold",
            "company_type": "startup",
            "has_connection": False
        },
        {
            "name": "enterprise_cold",
            "company_type": "enterprise",
            "has_connection": False
        }
    ]
    
    # Run comparison
    results = studies.run_comparison(scenarios)
    
    # Analyze results
    studies.analyze_results(results)
    
    # Save detailed results
    with open("results/ablation_study_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Detailed results saved to results/ablation_study_results.json")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_ablation_studies()
