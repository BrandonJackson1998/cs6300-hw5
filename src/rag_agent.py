"""
RAG Agent for BoardGameGeek Question Answering

This module handles:
1. Loading the persisted ChromaDB vector store
2. Processing user queries with optional LLM preprocessing
3. Retrieving relevant game chunks
4. Generating grounded answers using Gemini
"""

import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

class QueryPreprocessor:
    """Uses LLM to extract structured filters from natural language queries"""
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found. Please add it to your .env file.")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,  # Low temperature for consistent extraction
            google_api_key=api_key
        )
    
    def preprocess(self, query: str) -> Dict[str, Any]:
        """
        Extract structured filters from natural language query using LLM
        
        Args:
            query: User's natural language question
            
        Returns:
            Dictionary of extracted filters
        """
        prompt = self._create_extraction_prompt(query)
        
        try:
            response = self.llm.invoke(prompt)
            filters = self._parse_llm_response(response.content)
            return filters
        except Exception as e:
            print(f"Warning: Query preprocessing failed: {e}")
            # Return empty filters on failure (fallback to normal retrieval)
            return self._empty_filters()
    
    def _create_extraction_prompt(self, query: str) -> str:
        """Create prompt for LLM to extract filters"""
        return f"""You are a board game query analyzer. Extract structured filters from the user's natural language query about board games.

Available mechanics (common examples):
- Worker Placement, Cooperative Game, Deck Building, Dice Rolling, Hand Management
- Area Control, Area Majority / Influence, Set Collection, Tile Placement
- Drafting, Engine Building, Push Your Luck, Trick-taking

Available themes (common examples):
- Fantasy, Science Fiction, Medieval, Horror, Zombies, War, Pirates
- Economic, Farming, Space Exploration, Ancient, Mythology

Available categories:
- Strategy, Party, Family, War, Abstract, Thematic, Childrens

Available filters to extract:
1. mechanics: List of mechanic names (exact match from examples above)
2. themes: List of theme names
3. categories: List of category names
4. min_players: Minimum player count (integer)
5. max_players: Maximum player count (integer)
6. min_rating: Minimum rating threshold (float 0-10) - use for "rated above X", "8.5 and above", etc.
7. max_rating: Maximum rating threshold (float 0-10) - use for "rated below X", "under 7.0", etc.
8. min_playtime: Minimum playtime in minutes (integer)
9. max_playtime: Maximum playtime in minutes (integer)
10. sort_by_rating: true if query asks for "best", "highest rated", "top rated", OR if rating filters are used

User Query: "{query}"

Instructions:
- Only extract filters that are CLEARLY implied by the query
- Use exact mechanic/theme names when possible
- If asking for "games like [GameName]", infer common mechanics/themes
- Set sort_by_rating to true whenever rating thresholds are specified (min_rating/max_rating)
- For "X and above" or "rated X+", use min_rating
- Return valid JSON only, no other text
- If no filters detected, return empty object {{}}

Return JSON:"""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        # Try to extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            filters = json.loads(json_str)
            
            # Validate and clean filters
            return self._validate_filters(filters)
        else:
            return self._empty_filters()
    
    def _validate_filters(self, filters: Dict) -> Dict[str, Any]:
        """Validate and normalize extracted filters"""
        validated = self._empty_filters()
        
        # Validate mechanics (ensure it's a list)
        if 'mechanics' in filters and filters['mechanics']:
            validated['mechanics'] = filters['mechanics'] if isinstance(filters['mechanics'], list) else [filters['mechanics']]
        
        # Validate themes
        if 'themes' in filters and filters['themes']:
            validated['themes'] = filters['themes'] if isinstance(filters['themes'], list) else [filters['themes']]
        
        # Validate categories
        if 'categories' in filters and filters['categories']:
            validated['categories'] = filters['categories'] if isinstance(filters['categories'], list) else [filters['categories']]
        
        # Validate numeric filters
        if 'min_players' in filters and filters['min_players'] is not None:
            validated['min_players'] = int(filters['min_players'])
        
        if 'max_players' in filters and filters['max_players'] is not None:
            validated['max_players'] = int(filters['max_players'])
        
        if 'min_rating' in filters and filters['min_rating'] is not None:
            validated['min_rating'] = float(filters['min_rating'])
            # For LLM extraction, default to exclusive (>) behavior unless specified otherwise
            validated['min_rating_inclusive'] = False
            # Auto-enable rating sort when rating filter is present
            validated['sort_by_rating'] = True
        
        if 'max_rating' in filters and filters['max_rating'] is not None:
            validated['max_rating'] = float(filters['max_rating'])
            # Auto-enable rating sort when rating filter is present
            validated['sort_by_rating'] = True
        
        if 'min_playtime' in filters and filters['min_playtime'] is not None:
            validated['min_playtime'] = int(filters['min_playtime'])
        
        if 'max_playtime' in filters and filters['max_playtime'] is not None:
            validated['max_playtime'] = int(filters['max_playtime'])
        
        if 'sort_by_rating' in filters:
            validated['sort_by_rating'] = bool(filters['sort_by_rating'])
        
        return validated
    
    def _empty_filters(self) -> Dict[str, Any]:
        """Return empty filter structure"""
        return {
            'mechanics': [],
            'themes': [],
            'categories': [],
            'min_players': None,
            'max_players': None,
            'min_playtime': None,
            'max_playtime': None,
            'min_rating': None,
            'max_rating': None,
            'min_rating_inclusive': False,  # Default to exclusive (>)
            'sort_by_rating': False
        }

class BoardGameRAG:
    """RAG Agent for BoardGameGeek queries"""
    
    def __init__(self, persist_dir: str = "chroma_db", top_k: int = 20, use_preprocessing: bool = True):
        """
        Initialize the RAG agent
        
        Args:
            persist_dir: Directory where ChromaDB is persisted
            top_k: Number of documents to retrieve
            use_preprocessing: Whether to use LLM-based query preprocessing
        """
        self.persist_dir = persist_dir
        self.top_k = top_k
        self.use_preprocessing = use_preprocessing
        
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
        
        # Add query preprocessor
        if self.use_preprocessing:
            print("Initializing query preprocessor...")
            self.preprocessor = QueryPreprocessor()
            print("✓ Query preprocessor initialized")
        
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
        Retrieve relevant documents for a query with optional LLM preprocessing
        
        Args:
            query: User question
            
        Returns:
            List of unique games (deduplicated by game name) with applied filters.
            Returns UP TO top_k results, or fewer if that's all that match.
        """
        # Step 1: Preprocess query if enabled
        if self.use_preprocessing:
            print("  Preprocessing query with LLM...")
            filters = self.preprocessor.preprocess(query)
            
            # Show extracted filters
            non_empty_filters = dict((k, v) for k, v in filters.items() if v)
            print(f"  Extracted filters: {non_empty_filters}")
            
            # Show how the query would be interpreted
            if non_empty_filters:
                interpretation_parts = []
                if filters.get('categories'):
                    interpretation_parts.append(f"Categories: {', '.join(filters['categories'])}")
                if filters.get('mechanics'):
                    interpretation_parts.append(f"Mechanics: {', '.join(filters['mechanics'])}")
                if filters.get('themes'):
                    interpretation_parts.append(f"Themes: {', '.join(filters['themes'])}")
                if filters.get('min_players') or filters.get('max_players'):
                    if filters.get('min_players') == filters.get('max_players'):
                        interpretation_parts.append(f"Players: exactly {filters['min_players']}")
                    else:
                        min_p = filters.get('min_players', 'any')
                        max_p = filters.get('max_players', 'any')
                        interpretation_parts.append(f"Players: {min_p}-{max_p}")
                if filters.get('min_rating'):
                    op = ">=" if filters.get('min_rating_inclusive') else ">"
                    interpretation_parts.append(f"Rating {op} {filters['min_rating']}")
                if filters.get('max_rating'):
                    interpretation_parts.append(f"Rating < {filters['max_rating']}")
                if filters.get('min_playtime'):
                    interpretation_parts.append(f"Playtime >= {filters['min_playtime']} min")
                if filters.get('max_playtime'):
                    interpretation_parts.append(f"Playtime <= {filters['max_playtime']} min")
                if filters.get('sort_by_rating'):
                    interpretation_parts.append("Sort by rating (highest first)")
                
                print(f"  LLM interpretation: {' | '.join(interpretation_parts)}")
        else:
            # Fallback to keyword-based extraction
            filters = self._extract_filters(query.lower())
        
        # Step 2: Retrieve large candidate set
        # Use more candidates for rating filtering since it's very restrictive
        has_rating_filter = filters['min_rating'] is not None or filters['max_rating'] is not None
        candidate_multiplier = 15 if has_rating_filter else (10 if filters['sort_by_rating'] else 5)
        candidates = self.vectorstore.similarity_search(
            query=query,
            k=self.top_k * candidate_multiplier
        )
        
        # Step 3: Apply filters
        filtered_docs = self._apply_filters(candidates, filters)
        
        # Step 4: Check if we have enough results, provide fallback if needed
        has_filters = any([
            filters['mechanics'], filters['themes'], filters['categories'],
            filters['min_players'] is not None, filters['max_players'] is not None,
            filters.get('min_rating'), filters.get('max_rating'),
            filters.get('min_playtime'), filters.get('max_playtime')
        ])
        
        if has_filters and len(filtered_docs) < self.top_k // 3:  # Less than 1/3 of desired results
            print(f"⚠️  Only found {len(filtered_docs)} games matching all filters.")
            print("   Consider broadening your search criteria.")
            
            # If we have very few results, fallback to unfiltered semantic search
            if len(filtered_docs) < 3:
                print("   Falling back to semantic search without strict filters...")
                filtered_docs = candidates  # Use unfiltered candidates
        
        # Step 5: Sort if needed
        if filters['sort_by_rating']:
            filtered_docs.sort(
                key=lambda doc: doc.metadata.get('bayes_avg_rating', 0),
                reverse=True
            )
        
        # Step 6: Deduplicate and return UP TO top_k
        result = self._deduplicate(filtered_docs, self.top_k)
        
        if has_filters and len(result) > 0:
            print(f"✓ Returning {len(result)} games (filtered from {len(candidates)} candidates)")
        
        return result
    
    def _extract_filters(self, query_lower: str) -> Dict[str, Any]:
        """Extract filters from query"""
        filters = {
            'mechanics': [],
            'themes': [],
            'categories': [],
            'min_players': None,
            'max_players': None,
            'min_playtime': None,
            'max_playtime': None,
            'min_rating': None,
            'max_rating': None,
            'min_rating_inclusive': False,  # True for "at least", False for "higher than/above/over"
            'sort_by_rating': False
        }
        
        # Detect mechanics
        mechanic_map = {
            'worker placement': 'Worker Placement',
            'deck building': 'Deck, Bag, and Pool Building',
            'deck construction': 'Deck Construction',
            'cooperative': 'Cooperative Game',
            'co-op': 'Cooperative Game',
            'dice rolling': 'Dice Rolling',
            'area control': 'Area Majority / Influence',
            'area majority': 'Area Majority / Influence',
            'hand management': 'Hand Management',
            'drafting': 'Drafting',
            'set collection': 'Set Collection',
            'tile placement': 'Tile Placement',
            'push your luck': 'Push Your Luck',
            'trick-taking': 'Trick-taking',
            'engine building': 'Engine Building',
            'auction': 'Auction/Bidding',
            'bidding': 'Auction/Bidding'
        }
        
        for keyword, mechanic in mechanic_map.items():
            if keyword in query_lower:
                filters['mechanics'].append(mechanic)
        
        # Detect themes
        theme_map = {
            'fantasy': 'Fantasy',
            'sci-fi': 'Science Fiction',
            'science fiction': 'Science Fiction',
            'medieval': 'Medieval',
            'horror': 'Horror',
            'zombies': 'Zombies',
            'war': 'War',
            'ww2': 'World War II',
            'world war': 'World War II',
            'pirates': 'Pirates',
            'space': 'Space Exploration',
            'economic': 'Economic',
            'farming': 'Farming',
            'ancient': 'Ancient',
            'mythology': 'Mythology'
        }
        
        for keyword, theme in theme_map.items():
            if keyword in query_lower:
                filters['themes'].append(theme)
        
        # Detect categories
        category_map = {
            'strategy': 'Strategy',
            'party': 'Party',
            'family': 'Family',
            'war': 'War',
            'abstract': 'Abstract',
            'thematic': 'Thematic',
            'children': 'Childrens',
            'kids': 'Childrens'
        }
        
        for keyword, category in category_map.items():
            if keyword in query_lower:
                filters['categories'].append(category)
        
        # Detect player counts
        import re
        
        # Solo/1 player
        if any(term in query_lower for term in ['solo', '1 player', 'one player']):
            filters['min_players'] = 1
            filters['max_players'] = 1
        
        # 2 player
        elif any(term in query_lower for term in ['2 player', '2-player', 'two player']):
            filters['min_players'] = 2
            filters['max_players'] = 2
        
        # Large group patterns
        elif any(term in query_lower for term in ['large group', '6+ players', 'big group']):
            filters['min_players'] = 6
        
        # General number patterns
        else:
            player_match = re.search(r'(\d+)\s*(?:to|-)\s*(\d+)\s*players?', query_lower)
            if player_match:
                filters['min_players'] = int(player_match.group(1))
                filters['max_players'] = int(player_match.group(2))
            else:
                single_player_match = re.search(r'(\d+)\s*players?', query_lower)
                if single_player_match:
                    num = int(single_player_match.group(1))
                    filters['min_players'] = num
                    filters['max_players'] = num
        
        # Detect playtime filters
        if any(term in query_lower for term in ['quick', 'short', 'under 30']):
            filters['max_playtime'] = 30
        elif any(term in query_lower for term in ['long', 'over 120', 'lengthy']):
            filters['min_playtime'] = 120
        
        # Detect rating threshold patterns
        rating_pattern = r'(rated|rating)\s+(higher than|above|over|at least|below|under)\s+(\d+\.?\d*)'
        rating_match = re.search(rating_pattern, query_lower)
        
        # Also check for "rating of X or higher/better" patterns
        if not rating_match:
            alt_pattern = r'rating\s+of\s+(\d+\.?\d*)\s+or\s+(higher|better)'
            alt_match = re.search(alt_pattern, query_lower)
            if alt_match:
                threshold = float(alt_match.group(1))
                filters['min_rating'] = threshold
                filters['min_rating_inclusive'] = True  # "X or higher" is inclusive
                filters['sort_by_rating'] = True
        
        if rating_match:
            comparator = rating_match.group(2)
            threshold = float(rating_match.group(3))
            
            if comparator in ['higher than', 'above', 'over']:
                filters['min_rating'] = threshold
                filters['min_rating_inclusive'] = False  # Use > (exclusive)
                filters['sort_by_rating'] = True  # Sort by rating when filtering
            elif comparator == 'at least':
                filters['min_rating'] = threshold  
                filters['min_rating_inclusive'] = True  # Use >= (inclusive)
                filters['sort_by_rating'] = True
            elif comparator in ['below', 'under']:
                filters['max_rating'] = threshold
                filters['sort_by_rating'] = True
        
        # Detect general rating queries
        if any(term in query_lower for term in ['highest rated', 'best rated', 'top rated', 'best', 'highest ranking', 'top']):
            filters['sort_by_rating'] = True
        
        return filters
    
    def _apply_filters(self, candidates: List[Document], filters: Dict) -> List[Document]:
        """Apply metadata filters to candidates"""
        filtered = candidates
        
        # Filter by mechanics
        if filters['mechanics']:
            filtered = [
                doc for doc in filtered
                if any(mech in doc.metadata.get('mechanics', '') for mech in filters['mechanics'])
            ]
        
        # Filter by themes
        if filters['themes']:
            filtered = [
                doc for doc in filtered
                if any(theme in doc.metadata.get('themes', '') for theme in filters['themes'])
            ]
        
        # Filter by categories
        if filters['categories']:
            filtered = [
                doc for doc in filtered
                if any(cat in doc.metadata.get('categories', '') for cat in filters['categories'])
            ]
        
        # Filter by player count
        if filters['min_players'] is not None:
            if filters['max_players'] is not None:
                # Range specified
                filtered = [
                    doc for doc in filtered
                    if (doc.metadata.get('min_players', 999) <= filters['max_players'] and
                        doc.metadata.get('max_players', 0) >= filters['min_players'])
                ]
            else:
                # Minimum only
                filtered = [
                    doc for doc in filtered
                    if doc.metadata.get('max_players', 0) >= filters['min_players']
                ]
        
        # Filter by playtime
        if filters.get('max_playtime'):
            filtered = [
                doc for doc in filtered
                if doc.metadata.get('mfg_playtime', 999) <= filters['max_playtime']
            ]
        
        if filters.get('min_playtime'):
            filtered = [
                doc for doc in filtered
                if doc.metadata.get('mfg_playtime', 0) >= filters['min_playtime']
            ]
        
        # Filter by minimum rating
        if filters['min_rating'] is not None:
            if filters['min_rating_inclusive']:
                # "at least" - use >= (inclusive)
                filtered = [
                    doc for doc in filtered
                    if doc.metadata.get('bayes_avg_rating', 0) >= filters['min_rating']
                ]
            else:
                # "higher than", "above", "over" - use > (exclusive)
                filtered = [
                    doc for doc in filtered
                    if doc.metadata.get('bayes_avg_rating', 0) > filters['min_rating']
                ]
        
        # Filter by maximum rating
        if filters['max_rating'] is not None:
            filtered = [
                doc for doc in filtered
                if doc.metadata.get('bayes_avg_rating', 0) < filters['max_rating']
            ]
        
        return filtered
    
    def _deduplicate(self, docs: List[Document], k: int) -> List[Document]:
        """
        Deduplicate by game name and return UP TO k results
        
        Args:
            docs: List of documents to deduplicate
            k: Maximum number of unique games to return
            
        Returns:
            List of unique documents (by game name), up to k results.
            If fewer than k unique games exist, returns all unique games found.
        """
        seen_games = set()
        unique_docs = []
        
        for doc in docs:
            game_name = doc.metadata['name']
            if game_name not in seen_games:
                seen_games.add(game_name)
                unique_docs.append(doc)
                
                # Exit early if we have enough results
                if len(unique_docs) >= k:
                    break
        
        # Returns whatever we found, even if < k
        return unique_docs
    
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
- Playtime: {meta['mfg_playtime']} min
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
            if self.use_preprocessing:
                print("(Using LLM query preprocessing)")
            print(f"{'='*80}\n")
        
        # Step 1: Retrieve
        if verbose:
            print(f"Retrieving top {self.top_k} relevant games...")
        documents = self.retrieve(question)
        
        # Handle case where no documents were retrieved
        if not documents:
            error_msg = "No games found matching your criteria. Try a broader search."
            if verbose:
                print(f"❌ {error_msg}")
            return {
                'question': question,
                'answer': error_msg,
                'retrieved_documents': [],
                'num_retrieved': 0
            }
        
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
    
    # Let user choose preprocessing option
    use_preprocessing = input("Use LLM query preprocessing? (y/n, default=y): ").strip().lower()
    use_preprocessing = use_preprocessing != 'n'
    
    # Initialize RAG agent
    rag = BoardGameRAG(top_k=20, use_preprocessing=use_preprocessing)
    
    # Example queries
    example_queries = [
        "What are good strategy games for 2 players?",
        "Recommend cooperative games for families",
        "What games have worker placement mechanics?",
        "Find party games that play in under 30 minutes",
        "Best rated games like Catan but shorter",
        "Fun cooperative games for family game night with 4 people"
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