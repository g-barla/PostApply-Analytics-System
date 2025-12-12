"""
Career Q&A Chain
Pure RAG system for answering career-related questions from knowledge base
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advanced_rag_system import AdvancedRAGSystem


class CareerQAChain:
    """
    Simple Q&A chain using pure RAG (no RL context needed)
    """
    
    def __init__(self):
        self.rag = AdvancedRAGSystem()
    
    def ask(self, question):
        """
        Answer any career-related question using knowledge base
        
        Args:
            question: str, any career question
        
        Returns:
            Dict with answer and sources
        """
        
        print(f"\n{'='*70}")
        print(f"CAREER Q&A CHAIN")
        print(f"{'='*70}")
        print(f"Question: {question}")
        print(f"{'='*70}\n")
        
        print(f"🔍 Searching knowledge base...")
        result = self.rag.query(question, k=3)
        
        print(f"✅ Found {len(result['sources'])} relevant sources")
        print(f"\nSOURCES:")
        for source in result['sources']:
            print(f"  • {source['filename']} (category: {source['category']})")
        
        print(f"\nANSWER:")
        print(result['answer'])
        
        return {
            "question": question,
            "answer": result['answer'],
            "sources": result['sources']
        }


def test_career_qa():
    """Test the career Q&A chain"""
    
    qa = CareerQAChain()
    
    questions = [
        "How do I prepare for a data analyst interview?",
        "What should I include in my follow-up email?",
        "How can I find a hiring manager's email?",
        "What are some SQL questions I should prepare for?",
        "How do I research company culture?"
    ]
    
    results = {}
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(questions)}")
        print(f"{'='*70}")
        
        result = qa.ask(question)
        
        results[f"question_{i}"] = {
            "question": result['question'],
            "answer": result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'],
            "sources": [s['filename'] for s in result['sources']]
        }
    
    # Save results
    with open("results/career_qa_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to results/career_qa_tests.json")
    print("="*70)


if __name__ == "__main__":
    test_career_qa()
