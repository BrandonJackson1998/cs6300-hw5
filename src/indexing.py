"""
Indexing Pipeline for BoardGameGeek RAG System

This module handles:
1. Loading and merging game data with mechanics, themes, and subcategories
2. Creating chunks (1 game = 1 chunk)
3. Generating embeddings using Google's embedding model
4. Storing in ChromaDB with metadata
"""

import pandas as pd
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

class BoardGameIndexer:
    """Handles indexing of BoardGameGeek data into ChromaDB"""
    
    def __init__(self, data_dir: str = "data", persist_dir: str = "chroma_db"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all CSV files"""
        print("Loading CSV files...")
        games = pd.read_csv(os.path.join(self.data_dir, 'games.csv'))
        mechanics = pd.read_csv(os.path.join(self.data_dir, 'mechanics.csv'))
        themes = pd.read_csv(os.path.join(self.data_dir, 'themes.csv'))
        subcategories = pd.read_csv(os.path.join(self.data_dir, 'subcategories.csv'))
        
        print(f"✓ Loaded {len(games)} games")
        print(f"✓ Loaded {len(mechanics)} mechanics")
        print(f"✓ Loaded {len(themes)} themes")
        print(f"✓ Loaded {len(subcategories)} subcategories")
        
        return games, mechanics, themes, subcategories
    
    def get_active_features(self, row: pd.Series, feature_cols: List[str]) -> List[str]:
        """Extract list of active features from one-hot encoded row"""
        return [col for col in feature_cols if row[col] == 1]
    
    def create_documents(self, games: pd.DataFrame, mechanics: pd.DataFrame, 
                        themes: pd.DataFrame, subcategories: pd.DataFrame) -> List[Document]:
        """
        Create LangChain Document objects from game data
        
        Each document contains:
        - page_content: The game description (to be embedded)
        - metadata: All other game information
        """
        print("\nCreating documents...")
        documents = []
        
        # Get feature column names
        mechanic_cols = [col for col in mechanics.columns if col != 'BGGId']
        theme_cols = [col for col in themes.columns if col != 'BGGId']
        subcat_cols = [col for col in subcategories.columns if col != 'BGGId']
        
        # Filter out games without descriptions
        games_with_desc = games[games['Description'].notna()].copy()
        print(f"Processing {len(games_with_desc)} games with descriptions...")
        
        for idx, game in games_with_desc.iterrows():
            bgg_id = game['BGGId']
            
            # Get mechanics, themes, subcategories for this game
            game_mechs = mechanics[mechanics['BGGId'] == bgg_id].iloc[0]
            game_themes = themes[themes['BGGId'] == bgg_id].iloc[0]
            game_subcats = subcategories[subcategories['BGGId'] == bgg_id].iloc[0]
            
            # Extract active features
            active_mechanics = self.get_active_features(game_mechs, mechanic_cols)
            active_themes = self.get_active_features(game_themes, theme_cols)
            active_subcats = self.get_active_features(game_subcats, subcat_cols)
            
            # Create metadata dictionary
            metadata = {
                'bgg_id': int(bgg_id),
                'name': str(game['Name']),
                'year_published': int(game['YearPublished']),
                'min_players': int(game['MinPlayers']),
                'max_players': int(game['MaxPlayers']),
                'playtime': int(game['MfgPlaytime']),
                'min_age': int(game['MfgAgeRec']),
                'avg_rating': float(game['AvgRating']),
                'game_weight': float(game['GameWeight']),
                'num_ratings': int(game['NumUserRatings']),
                'rank': int(game['Rank:boardgame']) if game['Rank:boardgame'] != 21926 else None,
                'mechanics': ', '.join(active_mechanics) if active_mechanics else 'None',
                'themes': ', '.join(active_themes) if active_themes else 'None',
                'subcategories': ', '.join(active_subcats) if active_subcats else 'None',
            }
            
            # Create document
            doc = Document(
                page_content=game['Description'],
                metadata=metadata
            )
            documents.append(doc)
            
            # Progress indicator
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1} games...")
        
        print(f"✓ Created {len(documents)} documents")
        return documents
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create and persist ChromaDB vector store"""
        print("\nCreating vector store...")
        print("This may take several minutes for 20k+ documents...")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="boardgames"
        )
        
        print(f"✓ Vector store created and persisted to {self.persist_dir}")
        return vectorstore
    
    def index(self):
        """Main indexing pipeline"""
        print("="*80)
        print("BOARDGAMEGEEK INDEXING PIPELINE")
        print("="*80)
        
        # Step 1: Load data
        games, mechanics, themes, subcategories = self.load_data()
        
        # Step 2: Create documents
        documents = self.create_documents(games, mechanics, themes, subcategories)
        
        # Step 3: Create vector store
        vectorstore = self.create_vector_store(documents)
        
        print("\n" + "="*80)
        print("INDEXING COMPLETE")
        print("="*80)
        print(f"Total documents indexed: {len(documents)}")
        print(f"Vector store location: {self.persist_dir}")
        
        return vectorstore

def main():
    """Run the indexing pipeline"""
    indexer = BoardGameIndexer()
    indexer.index()

if __name__ == "__main__":
    main()