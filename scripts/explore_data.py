# scripts/explore_data.py
import pandas as pd
import os

def load_data():
    """Load all CSV files"""
    data_dir = 'data'
    
    print("Loading CSV files...")
    games = pd.read_csv(os.path.join(data_dir, 'games.csv'))
    mechanics = pd.read_csv(os.path.join(data_dir, 'mechanics.csv'))
    themes = pd.read_csv(os.path.join(data_dir, 'themes.csv'))
    subcategories = pd.read_csv(os.path.join(data_dir, 'subcategories.csv'))
    
    print(f"✓ Loaded {len(games)} games")
    print(f"✓ Loaded {len(mechanics)} mechanics")
    print(f"✓ Loaded {len(themes)} themes")
    print(f"✓ Loaded {len(subcategories)} subcategories")
    
    return games, mechanics, themes, subcategories

def explore_games_structure(games):
    """Explore the games.csv structure"""
    print("\n" + "="*80)
    print("GAMES.CSV STRUCTURE")
    print("="*80)
    
    print("\nColumns:")
    for i, col in enumerate(games.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nTotal games: {len(games)}")
    print(f"Shape: {games.shape}")
    
    print("\nFirst game sample:")
    sample = games.iloc[0]
    print(f"  Name: {sample['Name']}")
    print(f"  Year: {sample['YearPublished']}")
    print(f"  Rating: {sample['AvgRating']}")
    print(f"  Players: {sample['MinPlayers']}-{sample['MaxPlayers']}")
    print(f"  Playtime: {sample['MfgPlaytime']} min")
    print(f"  Weight: {sample['GameWeight']}")
    
    print("\nMissing values in key columns:")
    key_columns = ['Name', 'Description', 'YearPublished', 'AvgRating']
    for col in key_columns:
        if col in games.columns:
            missing = games[col].isna().sum()
            print(f"  {col}: {missing} ({missing/len(games)*100:.1f}%)")

def explore_mechanics_structure(mechanics):
    """Explore mechanics one-hot encoding"""
    print("\n" + "="*80)
    print("MECHANICS STRUCTURE (One-Hot Encoded)")
    print("="*80)
    
    # Get mechanic columns (exclude BGGId)
    mechanic_cols = [col for col in mechanics.columns if col != 'BGGId']
    print(f"\nTotal mechanics: {len(mechanic_cols)}")
    
    # Show most common mechanics
    mechanic_counts = mechanics[mechanic_cols].sum().sort_values(ascending=False)
    print("\nTop 15 most common mechanics:")
    for i, (mech, count) in enumerate(mechanic_counts.head(15).items(), 1):
        print(f"  {i:2d}. {mech}: {count} games ({count/len(mechanics)*100:.1f}%)")
    
    # Example game with mechanics
    sample_game_idx = 0
    sample_mechs = mechanics.iloc[sample_game_idx]
    active_mechs = [col for col in mechanic_cols if sample_mechs[col] == 1]
    print(f"\nExample: Game BGGId={sample_mechs['BGGId']} has {len(active_mechs)} mechanics:")
    for mech in active_mechs[:10]:  # Show first 10
        print(f"  - {mech}")

def explore_themes_structure(themes):
    """Explore themes one-hot encoding"""
    print("\n" + "="*80)
    print("THEMES STRUCTURE (One-Hot Encoded)")
    print("="*80)
    
    theme_cols = [col for col in themes.columns if col != 'BGGId']
    print(f"\nTotal themes: {len(theme_cols)}")
    
    theme_counts = themes[theme_cols].sum().sort_values(ascending=False)
    print("\nTop 15 most common themes:")
    for i, (theme, count) in enumerate(theme_counts.head(15).items(), 1):
        print(f"  {i:2d}. {theme}: {count} games ({count/len(themes)*100:.1f}%)")

def explore_subcategories_structure(subcategories):
    """Explore subcategories one-hot encoding"""
    print("\n" + "="*80)
    print("SUBCATEGORIES STRUCTURE (One-Hot Encoded)")
    print("="*80)
    
    subcat_cols = [col for col in subcategories.columns if col != 'BGGId']
    print(f"\nTotal subcategories: {len(subcat_cols)}")
    
    subcat_counts = subcategories[subcat_cols].sum().sort_values(ascending=False)
    print("\nAll subcategories:")
    for i, (subcat, count) in enumerate(subcat_counts.items(), 1):
        print(f"  {i:2d}. {subcat}: {count} games ({count/len(subcategories)*100:.1f}%)")

def analyze_descriptions(games):
    """Analyze description field for embedding"""
    print("\n" + "="*80)
    print("DESCRIPTION ANALYSIS (for embedding)")
    print("="*80)
    
    # Filter out missing descriptions
    games_with_desc = games[games['Description'].notna()]
    print(f"\nGames with descriptions: {len(games_with_desc)} / {len(games)}")
    
    # Description lengths
    desc_lengths = games_with_desc['Description'].str.len()
    print(f"\nDescription length statistics:")
    print(f"  Mean: {desc_lengths.mean():.0f} characters")
    print(f"  Median: {desc_lengths.median():.0f} characters")
    print(f"  Min: {desc_lengths.min():.0f} characters")
    print(f"  Max: {desc_lengths.max():.0f} characters")
    
    # Sample description
    print(f"\nSample description (first 500 chars):")
    sample_desc = games_with_desc.iloc[0]['Description']
    print(f"Game: {games_with_desc.iloc[0]['Name']}")
    print(f"{sample_desc[:500]}...")

def analyze_merged_data(games, mechanics, themes, subcategories):
    """Show how we'll merge the data"""
    print("\n" + "="*80)
    print("DATA MERGING STRATEGY")
    print("="*80)
    
    # Merge example
    sample_id = games.iloc[0]['BGGId']
    game_info = games[games['BGGId'] == sample_id].iloc[0]
    game_mechs = mechanics[mechanics['BGGId'] == sample_id].iloc[0]
    game_themes = themes[themes['BGGId'] == sample_id].iloc[0]
    game_subcats = subcategories[subcategories['BGGId'] == sample_id].iloc[0]
    
    print(f"\nExample Game: {game_info['Name']}")
    print(f"BGGId: {sample_id}")
    
    # Get active mechanics
    mech_cols = [col for col in mechanics.columns if col != 'BGGId']
    active_mechs = [col for col in mech_cols if game_mechs[col] == 1]
    print(f"\nMechanics ({len(active_mechs)}): {', '.join(active_mechs[:5])}...")
    
    # Get active themes
    theme_cols = [col for col in themes.columns if col != 'BGGId']
    active_themes = [col for col in theme_cols if game_themes[col] == 1]
    print(f"Themes ({len(active_themes)}): {', '.join(active_themes[:5])}...")
    
    # Get active subcategories
    subcat_cols = [col for col in subcategories.columns if col != 'BGGId']
    active_subcats = [col for col in subcat_cols if game_subcats[col] == 1]
    print(f"Subcategories ({len(active_subcats)}): {', '.join(active_subcats)}")

def main():
    print("\n" + "="*80)
    print("BOARDGAMEGEEK DATASET EXPLORATION")
    print("="*80)
    
    # Load data
    games, mechanics, themes, subcategories = load_data()
    
    # Explore structure
    explore_games_structure(games)
    explore_mechanics_structure(mechanics)
    explore_themes_structure(themes)
    explore_subcategories_structure(subcategories)
    analyze_descriptions(games)
    analyze_merged_data(games, mechanics, themes, subcategories)
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("\n✓ Mechanics, themes, and subcategories are ONE-HOT ENCODED")
    print("✓ Each game (row) has binary flags for each mechanic/theme/subcategory")
    print("✓ We can easily extract active mechanics/themes for metadata")
    print("\nEmbedding Strategy:")
    print("  - Embed: Description field")
    print("  - Metadata: Name, year, players, playtime, rating, weight,")
    print("              active mechanics, active themes, active subcategories")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("  1. Design PEAS framework")
    print("  2. Build indexing pipeline (merge data + create chunks)")
    print("  3. Set up ChromaDB vector store")
    print("  4. Implement retrieval + generation")
    print("  5. Evaluate with LLM-as-a-judge")

if __name__ == "__main__":
    main()