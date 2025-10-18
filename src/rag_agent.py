"""
RAG Agent for BoardGameGeek Question Answering

This module handles:
1. Loading the persisted ChromaDB vector store
2. Processing user queries
3. Retrieving relevant game chunks
4. Generating grounded answers using Gemini
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

class BoardGameRAG:
    """RAG Agent for BoardGameGeek queries"""
    
    def __init__(self, persist_dir: str = "chroma_db", top_k: int = 10):
        """
        Initialize the RAG agent
        
        Args:
            persist_dir: Directory where ChromaDB is persisted
            top_k: Number of documents to retrieve
        """
        self.persist_dir = persist_dir
        self.top_k = top_k
        
        # Initialize embeddings (same as indexing)
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vector store
        print("Loading vector store...")
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="boardgames"
        )
        
        # Initialize Gemini LLM
        print("Initializing Gemini LLM...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found. Please add it to your .env file.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            convert_system_message_to_human=True,
            google_api_key=api_key
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a board game expert assistant. Use the following game information to answer the question accurately.

Context (Retrieved Games):
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- Include specific game names when relevant
- Mention key metadata (player count, playtime, mechanics, themes) when helpful
- If the context doesn't contain enough information, say so
- Be concise but informative

Answer:"""
        )
        
        print("✓ RAG Agent initialized\n")
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User question
            
        Returns:
            List of relevant documents
        """
        results = self.vectorstore.similarity_search(
            query=query,
            k=self.top_k
        )
        return results
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            meta = doc.metadata
            context_part = f"""
Game {i}: {meta['name']} ({meta['year_published']})
- Rating: {meta['avg_rating']:.2f}/10 ({meta['num_ratings']} ratings)
- Players: {meta['min_players']}-{meta['max_players']}
- Playtime: {meta['playtime']} min
- Weight: {meta['game_weight']:.2f}/5
- Mechanics: {meta['mechanics']}
- Themes: {meta['themes']}
- Description: {doc.page_content[:300]}...
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate(self, query: str, context: str) -> str:
        """
        Generate answer using LLM
        
        Args:
            query: User question
            context: Formatted context from retrieval
            
        Returns:
            Generated answer
        """
        prompt = self.prompt_template.format(
            context=context,
            question=query
        )
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def query(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Main RAG pipeline: retrieve + generate
        
        Args:
            question: User question
            verbose: Whether to print intermediate steps
            
        Returns:
            Dictionary with answer and metadata
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"QUERY: {question}")
            print(f"{'='*80}\n")
        
        # Step 1: Retrieve
        if verbose:
            print(f"Retrieving top {self.top_k} relevant games...")
        documents = self.retrieve(question)
        
        if verbose:
            print(f"✓ Retrieved {len(documents)} games\n")
            print("Retrieved Games:")
            for i, doc in enumerate(documents, 1):
                print(f"  {i}. {doc.metadata['name']} (Rating: {doc.metadata['avg_rating']:.2f})")
        
        # Step 2: Format context
        context = self.format_context(documents)
        
        # Step 3: Generate answer
        if verbose:
            print(f"\nGenerating answer with Gemini...")
        answer = self.generate(question, context)
        
        if verbose:
            print(f"✓ Answer generated\n")
            print(f"{'='*80}")
            print("ANSWER:")
            print(f"{'='*80}")
            print(answer)
            print(f"{'='*80}\n")
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_documents': documents,
            'num_retrieved': len(documents)
        }

def main():
    """Interactive query interface"""
    print("="*80)
    print("BOARDGAMEGEEK RAG AGENT")
    print("="*80)
    
    # Initialize RAG agent
    rag = BoardGameRAG(top_k=10)
    
    # Example queries
    example_queries = [
        "What are good strategy games for 2 players?",
        "Recommend cooperative games for families",
        "What games have worker placement mechanics?",
        "Find party games that play in under 30 minutes"
    ]
    
    print("Example queries you can try:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "="*80)
    print("Enter your question (or 'quit' to exit)")
    print("="*80)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            rag.query(question, verbose=True)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()