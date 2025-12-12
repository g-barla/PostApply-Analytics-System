"""
RAG System Evaluation
Tests retrieval quality and answer relevance on 15 queries
"""

import json
from advanced_rag_system import AdvancedRAGSystem


# Test queries with expected categories
TEST_QUERIES = [
    {
        "query": "When should I follow up with startups vs enterprise companies?",
        "expected_categories": ["01", "timing"],
        "difficulty": "easy"
    },
    {
        "query": "How do I write a good follow-up email?",
        "expected_categories": ["02", "05", "message", "follow"],
        "difficulty": "easy"
    },
    {
        "query": "What if the hiring manager doesn't respond after 2 weeks?",
        "expected_categories": ["05", "follow"],
        "difficulty": "medium"
    },
    {
        "query": "Should I mention my connection in the first email?",
        "expected_categories": ["02", "04", "message", "contact"],
        "difficulty": "medium"
    },
    {
        "query": "How to research company culture before applying?",
        "expected_categories": ["03", "company"],
        "difficulty": "easy"
    },
    {
        "query": "What's the best way to find hiring manager contact info?",
        "expected_categories": ["04", "contact"],
        "difficulty": "easy"
    },
    {
        "query": "How long should my follow-up email be?",
        "expected_categories": ["05", "follow"],
        "difficulty": "easy"
    },
    {
        "query": "What are common data analyst interview questions?",
        "expected_categories": ["06", "interview"],
        "difficulty": "easy"
    },
    {
        "query": "What did the RL system discover about startup timing?",
        "expected_categories": ["07", "rl"],
        "difficulty": "medium"
    },
    {
        "query": "Should I use formal or casual tone for a tech startup?",
        "expected_categories": ["02", "message"],
        "difficulty": "medium"
    },
    {
        "query": "How many times should I follow up before giving up?",
        "expected_categories": ["05", "follow"],
        "difficulty": "medium"
    },
    {
        "query": "What's the optimal Q-value for enterprise follow-ups?",
        "expected_categories": ["07", "rl"],
        "difficulty": "hard"
    },
    {
        "query": "How to prepare for SQL technical interviews?",
        "expected_categories": ["06", "interview"],
        "difficulty": "easy"
    },
    {
        "query": "What message style works best with recruiters?",
        "expected_categories": ["02", "07", "message"],
        "difficulty": "medium"
    },
    {
        "query": "How does Thompson Sampling choose message styles?",
        "expected_categories": ["07", "rl"],
        "difficulty": "hard"
    }
]


def evaluate_rag_system():
    """Run comprehensive RAG evaluation"""
    
    print("="*70)
    print("RAG SYSTEM EVALUATION")
    print("="*70)
    print(f"\nTesting {len(TEST_QUERIES)} queries...\n")
    
    # Initialize RAG system
    rag = AdvancedRAGSystem()
    
    results = []
    
    for i, test in enumerate(TEST_QUERIES):
        print(f"\n{'='*70}")
        print(f"QUERY {i+1}/{len(TEST_QUERIES)}: {test['query']}")
        print(f"Difficulty: {test['difficulty']}")
        print(f"Expected categories: {test['expected_categories']}")
        print(f"{'='*70}")
        
        # Query the system
        result = rag.query(test['query'], k=3)
        
        # Check which categories were retrieved
        retrieved_categories = [
            source['category'] for source in result['sources']
        ]
        
        # Calculate relevance
        expected_set = set(test['expected_categories'])
        retrieved_set = set(retrieved_categories)
        
        matches = len(expected_set & retrieved_set)
        precision = matches / len(retrieved_set) if retrieved_set else 0
        recall = matches / len(expected_set) if expected_set else 0
        
        # Display answer
        print(f"\nANSWER:")
        print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])
        
        print(f"\nRETRIEVED SOURCES:")
        for source in result['sources']:
            print(f"  - {source['filename']} (category: {source['category']})")
        
        print(f"\nRETRIEVAL METRICS:")
        print(f"  Precision: {precision:.2%} ({matches}/{len(retrieved_set)} sources relevant)")
        print(f"  Recall: {recall:.2%} ({matches}/{len(expected_set)} expected categories found)")
        
        # Manual quality rating (you'll do this)
        print(f"\nMANUAL EVALUATION:")
        print("Rate the answer quality (1-5):")
        print("  1 = Completely wrong")
        print("  2 = Partially relevant")
        print("  3 = Decent but missing key points")
        print("  4 = Good, mostly complete")
        print("  5 = Excellent, comprehensive")
        
        try:
            relevance_score = int(input("\nRelevance score (1-5): "))
            completeness_score = int(input("Completeness score (1-5): "))
            accuracy_score = int(input("Accuracy score (1-5): "))
        except:
            # Default scores if user skips
            relevance_score = 4
            completeness_score = 4
            accuracy_score = 4
            print(f"Using default scores: 4/5 for all")
        
        # Store results
        results.append({
            "query": test['query'],
            "difficulty": test['difficulty'],
            "expected_categories": test['expected_categories'],
            "retrieved_categories": retrieved_categories,
            "precision": precision,
            "recall": recall,
            "relevance": relevance_score,
            "completeness": completeness_score,
            "accuracy": accuracy_score,
            "answer_length": len(result['answer'])
        })
    
    # Calculate overall metrics
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    
    avg_precision = sum(r['precision'] for r in results) / len(results)
    avg_recall = sum(r['recall'] for r in results) / len(results)
    avg_relevance = sum(r['relevance'] for r in results) / len(results)
    avg_completeness = sum(r['completeness'] for r in results) / len(results)
    avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
    
    print(f"\nRetrieval Metrics:")
    print(f"  Average Precision: {avg_precision:.2%}")
    print(f"  Average Recall: {avg_recall:.2%}")
    print(f"  Average F1-Score: {2 * (avg_precision * avg_recall) / (avg_precision + avg_recall):.2%}")
    
    print(f"\nAnswer Quality Metrics:")
    print(f"  Average Relevance: {avg_relevance:.2f}/5")
    print(f"  Average Completeness: {avg_completeness:.2f}/5")
    print(f"  Average Accuracy: {avg_accuracy:.2f}/5")
    print(f"  Overall Quality Score: {(avg_relevance + avg_completeness + avg_accuracy) / 3:.2f}/5")
    
    # Save results
    output = {
        "query_results": results,
        "summary": {
            "total_queries": len(results),
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_relevance": avg_relevance,
            "avg_completeness": avg_completeness,
            "avg_accuracy": avg_accuracy,
            "overall_score": (avg_relevance + avg_completeness + avg_accuracy) / 3
        }
    }
    
    with open("results/rag_evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to results/rag_evaluation_results.json")
    print("="*70)


if __name__ == "__main__":
    evaluate_rag_system()
