"""
Indexing Pipeline for BoardGameGeek RAG System

This module handles:
1. Loading and merging game data with mechanics, themes, subcategories, designers, publishers, artists
2. Creating 2 chunks per game:
   - Chunk 1: Name + Description (semantic)
   - Chunk 2: Name + Structured Metadata (feature matching)
3. Generating embeddings using local HuggingFace model
4. Storing in ChromaDB with metadata
"""

import pandas as pd
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

class BoardGameIndexer:
    """Handles indexing of BoardGameGeek data into ChromaDB"""
    
    def __init__(self, data_dir: str = "data", persist_dir: str = "chroma_db"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        
        # Use local HuggingFace embeddings - no API calls!
        print("Initializing local embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Fast, good quality, 384-dim embeddings
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Embedding model loaded")
        
    def load_data(self) -> tuple:
        """Load all CSV files"""
        print("\nLoading CSV files...")
        games = pd.read_csv(os.path.join(self.data_dir, 'games.csv'))
        mechanics = pd.read_csv(os.path.join(self.data_dir, 'mechanics.csv'))
        themes = pd.read_csv(os.path.join(self.data_dir, 'themes.csv'))
        subcategories = pd.read_csv(os.path.join(self.data_dir, 'subcategories.csv'))
        designers = pd.read_csv(os.path.join(self.data_dir, 'designers_reduced.csv'))
        publishers = pd.read_csv(os.path.join(self.data_dir, 'publishers_reduced.csv'))
        artists = pd.read_csv(os.path.join(self.data_dir, 'artists_reduced.csv'))
        
        print(f"✓ Loaded {len(games)} games")
        print(f"✓ Loaded {len(mechanics)} mechanics")
        print(f"✓ Loaded {len(themes)} themes")
        print(f"✓ Loaded {len(subcategories)} subcategories")
        print(f"✓ Loaded {len(designers)} designers")
        print(f"✓ Loaded {len(publishers)} publishers")
        print(f"✓ Loaded {len(artists)} artists")
        
        return games, mechanics, themes, subcategories, designers, publishers, artists
    
    def get_active_features(self, row: pd.Series, feature_cols: List[str]) -> List[str]:
        """Extract list of active features from one-hot encoded row"""
        return [col for col in feature_cols if row[col] == 1]
    
    def get_high_level_categories(self, game: pd.Series) -> List[str]:
        """Extract high-level categories (Cat:Strategy, Cat:Thematic, etc.)"""
        cat_columns = [
            'Cat:Thematic', 'Cat:Strategy', 'Cat:War', 'Cat:Family',
            'Cat:CGS', 'Cat:Abstract', 'Cat:Party', 'Cat:Childrens'
        ]
        categories = []
        for cat in cat_columns:
            if cat in game and game[cat] == 1:
                # Remove 'Cat:' prefix for cleaner display
                categories.append(cat.replace('Cat:', ''))
        return categories
    
    def create_documents(self, games: pd.DataFrame, mechanics: pd.DataFrame, 
                        themes: pd.DataFrame, subcategories: pd.DataFrame,
                        designers: pd.DataFrame, publishers: pd.DataFrame,
                        artists: pd.DataFrame) -> List[Document]:
        """
        Create LangChain Document objects from game data
        
        Each game creates 2 documents:
        1. Name + Description (semantic search)
        2. Name + Structured Metadata (feature matching)
        """
        print("\nCreating documents...")
        documents = []
        
        # Get feature column names
        mechanic_cols = [col for col in mechanics.columns if col != 'BGGId']
        theme_cols = [col for col in themes.columns if col != 'BGGId']
        subcat_cols = [col for col in subcategories.columns if col != 'BGGId']
        designer_cols = [col for col in designers.columns if col not in ['BGGId', 'Low-Exp Designer']]
        publisher_cols = [col for col in publishers.columns if col not in ['BGGId', 'Low-Exp Publisher']]
        artist_cols = [col for col in artists.columns if col not in ['BGGId', 'Low-Exp Artist']]
        
        # Filter out games without descriptions
        games_with_desc = games[games['Description'].notna()].copy()
        print(f"Processing {len(games_with_desc)} games with descriptions...")
        
        for idx, game in games_with_desc.iterrows():
            bgg_id = game['BGGId']
            
            # Get features for this game
            game_mechs = mechanics[mechanics['BGGId'] == bgg_id].iloc[0]
            game_themes = themes[themes['BGGId'] == bgg_id].iloc[0]
            game_subcats = subcategories[subcategories['BGGId'] == bgg_id].iloc[0]
            game_designers = designers[designers['BGGId'] == bgg_id].iloc[0]
            game_publishers = publishers[publishers['BGGId'] == bgg_id].iloc[0]
            game_artists = artists[artists['BGGId'] == bgg_id].iloc[0]
            
            # Extract active features
            active_mechanics = self.get_active_features(game_mechs, mechanic_cols)
            active_themes = self.get_active_features(game_themes, theme_cols)
            active_subcats = self.get_active_features(game_subcats, subcat_cols)
            active_designers = self.get_active_features(game_designers, designer_cols)
            active_publishers = self.get_active_features(game_publishers, publisher_cols)
            active_artists = self.get_active_features(game_artists, artist_cols)
            high_level_cats = self.get_high_level_categories(game)
            
            # Create shared metadata dictionary
            metadata = {
                'bgg_id': int(bgg_id),
                'name': str(game['Name']),
                'year_published': int(game['YearPublished']),
                'min_players': int(game['MinPlayers']),
                'max_players': int(game['MaxPlayers']),
                'best_players': int(game['BestPlayers']),
                'mfg_playtime': int(game['MfgPlaytime']),
                'com_min_playtime': int(game['ComMinPlaytime']),
                'com_max_playtime': int(game['ComMaxPlaytime']),
                'min_age': int(game['MfgAgeRec']),
                'avg_rating': float(game['AvgRating']),
                'bayes_avg_rating': float(game['BayesAvgRating']),
                'game_weight': float(game['GameWeight']),
                'num_ratings': int(game['NumUserRatings']),
                'mechanics': ', '.join(active_mechanics) if active_mechanics else 'None',
                'themes': ', '.join(active_themes) if active_themes else 'None',
                'subcategories': ', '.join(active_subcats) if active_subcats else 'None',
                'categories': ', '.join(high_level_cats) if high_level_cats else 'None',
                'designers': ', '.join(active_designers) if active_designers else 'Unknown',
                'publishers': ', '.join(active_publishers) if active_publishers else 'Unknown',
                'artists': ', '.join(active_artists) if active_artists else 'Unknown',
            }
            
            # Add optional fields only if they're not None (ChromaDB requirement)
            if pd.notna(game['ComAgeRec']):
                metadata['com_age_rec'] = float(game['ComAgeRec'])
            
            if game['Rank:boardgame'] != 21926 and pd.notna(game['Rank:boardgame']):
                metadata['rank'] = int(game['Rank:boardgame'])
            
            # Final safety check: Filter out any remaining None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            # CHUNK 1: Name + Description (Semantic)
            description_content = f"{game['Name']}\n\n{game['Description']}"
            
            doc1 = Document(
                page_content=description_content,
                metadata={**metadata, 'chunk_type': 'description'}
            )
            documents.append(doc1)
            
            # CHUNK 2: Name + Structured Metadata (Feature Matching)
            metadata_content = f"""{game['Name']}

Rating: {game['AvgRating']:.2f}/10 (Bayes: {game['BayesAvgRating']:.2f}, {game['NumUserRatings']} ratings)
Mechanics: {', '.join(active_mechanics) if active_mechanics else 'None'}
Themes: {', '.join(active_themes) if active_themes else 'None'}
Subcategories: {', '.join(active_subcats) if active_subcats else 'None'}
Categories: {', '.join(high_level_cats) if high_level_cats else 'None'}
Players: {game['MinPlayers']}-{game['MaxPlayers']}
Best Player Count: {game['BestPlayers']}
Playtime: Manufacturer {game['MfgPlaytime']} min, Community {game['ComMinPlaytime']}-{game['ComMaxPlaytime']} min
Weight: {game['GameWeight']:.2f}/5
Age Recommendation: Manufacturer {game['MfgAgeRec']}+, Community {game['ComAgeRec']:.1f}+
Year Published: {game['YearPublished']}
Designer: {', '.join(active_designers) if active_designers else 'Unknown'}
Publisher: {', '.join(active_publishers) if active_publishers else 'Unknown'}
Artist: {', '.join(active_artists) if active_artists else 'Unknown'}""".strip()
            
            doc2 = Document(
                page_content=metadata_content,
                metadata={**metadata, 'chunk_type': 'metadata'}
            )
            documents.append(doc2)
            
            # Progress indicator
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1} games ({len(documents)} chunks)...")
        
        print(f"✓ Created {len(documents)} documents ({len(documents)//2} games × 2 chunks)")
        return documents
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create and persist ChromaDB vector store"""
        print("\nCreating vector store...")
        print("Embedding documents locally (this may take several minutes)...")
        
        # Create vector store with batching for efficiency
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
        print("BOARDGAMEGEEK INDEXING PIPELINE (TWO-CHUNK STRATEGY)")
        print("="*80)
        
        # Step 1: Load data
        games, mechanics, themes, subcategories, designers, publishers, artists = self.load_data()
        
        # Step 2: Create documents (2 chunks per game)
        documents = self.create_documents(
            games, mechanics, themes, subcategories,
            designers, publishers, artists
        )
        
        # Step 3: Create vector store
        vectorstore = self.create_vector_store(documents)
        
        print("\n" + "="*80)
        print("INDEXING COMPLETE")
        print("="*80)
        print(f"Total documents indexed: {len(documents)}")
        print(f"Total games: {len(documents)//2}")
        print(f"Chunks per game: 2 (description + metadata)")
        print(f"Vector store location: {self.persist_dir}")
        print(f"Embedding model: all-MiniLM-L6-v2 (local)")
        
        return vectorstore

def main():
    """Run the indexing pipeline"""
    indexer = BoardGameIndexer()
    indexer.index()

if __name__ == "__main__":
    main()