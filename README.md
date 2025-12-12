# PostApply Analytics Platform

> Hybrid Reinforcement Learning + Generative AI system for job application optimization



## 🎯 Overview

PostApply combines reinforcement learning with retrieval-augmented generation to optimize job application follow-up strategies, achieving **20.6% improvement** in response rates.

**Key Results:**
- Response Rate: 32.0% → 38.6% (+20.6%)
- Interview Rate: 9.4% → 11.6% (+23.4%)
- Statistical Significance: p < 0.0001

## 🏗️ Architecture

**Three AI Technologies:**
1. **Reinforcement Learning** - Learns optimal timing (Q-Learning) and messaging style (Thompson Sampling)
2. **RAG** - Retrieval-augmented generation with 12K+ word career guidance knowledge base
3. **Prompt Engineering** - Specialized chains synthesizing RL + RAG insights

## 📊 Components

### 1. RL Intelligence Layer 
- **Q-Learning** for follow-up timing optimization (24 states, 6 actions)
- **Thompson Sampling** for message style selection (24 contexts, 3 styles)
- Trained on 500 simulated job applications
- Converged after ~300 episodes

### 2. RAG Career Guidance 
- 7-document knowledge base (12,470 words)
- 215 text chunks with semantic embeddings
- OpenAI text-embedding-3-small
- 70% retrieval precision, 3
