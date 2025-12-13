"""
Advanced RAG System for PostApply Career Guidance
Simplified version using FAISS instead of ChromaDB (more reliable)
"""

import os
import json
import requests
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import pickle

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


class AdvancedRAGSystem:
    """RAG system with FAISS and direct OpenAI API calls"""
    
    def __init__(self, knowledge_base_path=None, api_key=None):
        """Initialize RAG system"""
        
        print("🚀 Initializing Advanced RAG System...")
        
        # Get OpenAI API key
        self.api_key = api_key or openai.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        # If no path provided, use absolute path
        if knowledge_base_path is None:
           import os
           script_dir = os.path.dirname(os.path.abspath(__file__))
           knowledge_base_path = os.path.join(script_dir, "knowledge_base")
        print(f"   Knowledge base path: {knowledge_base_path}")
        self.knowledge_base_path = knowledge_base_path
        
        # OpenAI API endpoints
        self.embedding_url = "https://api.openai.com/v1/embeddings"
        self.chat_url = "https://api.openai.com/v1/chat/completions"
        
        # Load documents
        print("📚 Loading knowledge base...")
        self.documents = self._load_documents()
        print(f"   Loaded {len(self.documents)} documents")
        
        # Create chunks
        print("✂️  Creating chunks...")
        self.chunks = self._create_chunks()
        print(f"   Created {len(self.chunks)} chunks")
        
        # Build vector store
        print("🔮 Building vector store (takes 1-2 min)...")
        self._build_vectorstore()
        print("✅ RAG System ready!\n")
        
        self.query_logs = []
    
    def _load_documents(self) -> List[Document]:
        """Load text documents from knowledge base"""
        documents = []
        kb_path = Path(self.knowledge_base_path)
        
        for txt_file in sorted(kb_path.glob("*.txt")):
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(txt_file),
                    "filename": txt_file.name,
                    "category": txt_file.stem.split('_')[0]
                }
            )
            documents.append(doc)
        
        return documents
    
    def _create_chunks(self) -> List[Document]:
        """Split documents into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        chunks = splitter.split_documents(self.documents)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
        
        return chunks
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding via OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "input": text,
            "model": "text-embedding-3-small"
        }
        
        response = requests.post(self.embedding_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['data'][0]['embedding']
    
    def _build_vectorstore(self):
        """Build simple vector store with embeddings"""
        
        # Get embeddings for all chunks
        self.embeddings = []
        for i, chunk in enumerate(self.chunks):
            if i % 10 == 0:
                print(f"   Processing chunk {i+1}/{len(self.chunks)}...")
            embedding = self._get_embedding(chunk.page_content)
            self.embeddings.append(embedding)
        
        self.embeddings = np.array(self.embeddings)
        
        # Save for future use
        os.makedirs("./vector_db", exist_ok=True)
        with open("./vector_db/embeddings.pkl", "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "chunks": self.chunks
            }, f)
    
    def _similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Find most similar chunks to query"""
        
        # Get query embedding
        query_embedding = np.array(self._get_embedding(query))
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return top k chunks
        return [self.chunks[i] for i in top_k_indices]
    
    def _call_llm(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """Call OpenAI Chat API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000
        }
        
        response = requests.post(self.chat_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def query(self, question: str, rl_context: Dict = None, k: int = 3) -> Dict:
        """Query the RAG system"""
        
        print(f"\n🔍 Query: '{question}'")
        
        # Retrieve relevant docs
        docs = self._similarity_search(question, k=k)
        
        # Build context
        context = "\n\n".join([
            f"[{doc.metadata['filename']}]\n{doc.page_content}"
            for doc in docs
        ])
        
        # Build prompt
        if rl_context:
            system_prompt = f"""You are an expert career coach specializing in job search strategies.

RL System Recommendations:
- Timing: Wait {rl_context.get('wait_days', 'N/A')} days
- Style: {rl_context.get('style', 'N/A')}
- Confidence: {rl_context.get('confidence', 'N/A')}%
- Q-value: {rl_context.get('q_value', 'N/A')}

Synthesize the knowledge base guidance WITH the RL recommendations. Explain how they align.

Context from knowledge base:
{context}"""
        else:
            system_prompt = f"""You are an expert career coach specializing in job search strategies.

Use this context to answer the question. Be specific and actionable.

Context:
{context}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        print("   Generating answer...")
        answer = self._call_llm(messages)
        
        sources = [
            {
                "filename": doc.metadata['filename'],
                "category": doc.metadata.get('category', 'unknown'),
                "preview": doc.page_content[:150] + "..."
            }
            for doc in docs
        ]
        
        self.query_logs.append({
            "question": question,
            "rl_context": rl_context,
            "num_sources": len(docs)
        })
        
        print("✅ Done!\n")
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def save_logs(self, filepath: str = "results/rag_query_logs.json"):
        """Save query logs"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.query_logs, f, indent=2)
        print(f"💾 Logs saved to {filepath}")


# Test function
if __name__ == "__main__":
    print("="*60)
    print("TESTING RAG SYSTEM")
    print("="*60 + "\n")
    
    # Initialize
    rag = AdvancedRAGSystem()
    
    # Test query 1
    print("\n" + "="*60)
    print("TEST 1: Basic Query")
    print("="*60)
    
    result1 = rag.query("When should I follow up with a startup?")
    print("\nANSWER:")
    print(result1['answer'])
    print("\nSOURCES:", [s['filename'] for s in result1['sources']])
    
    # Test query 2 with RL context
    print("\n" + "="*60)
    print("TEST 2: Query with RL Context")
    print("="*60)
    
    rl_context = {
        "wait_days": 3,
        "style": "casual",
        "confidence": 85,
        "q_value": 8.42
    }
    
    result2 = rag.query(
        "Why should I wait 3 days for this startup application?",
        rl_context=rl_context
    )
    print("\nANSWER:")
    print(result2['answer'])
    print("\nSOURCES:", [s['filename'] for s in result2['sources']])
    
    # Save logs
    rag.save_logs()
    
    print("\n" + "="*60)
    print("✅ RAG SYSTEM TESTS COMPLETE!")
    print("="*60)
