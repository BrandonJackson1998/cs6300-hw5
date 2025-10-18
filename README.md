# BoardGameGeek RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system for querying BoardGameGeek data using natural language. The system combines semantic search with intelligent filtering and LLM-powered query preprocessing.

## 🎯 Features

- **Intelligent Query Processing**: LLM-based preprocessing extracts structured filters from natural language queries
- **Advanced Filtering**: Support for mechanics, themes, categories, player counts, ratings, and playtime
- **Local Embeddings**: Uses HuggingFace embeddings (no API quota limits)
- **Rating Threshold Filtering**: Precise numeric filtering for "games rated above 8.5"
- **Edge Case Handling**: Graceful fallbacks when filters are too restrictive
- **Interactive Interface**: Command-line chat interface with example queries

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and enter the repository
git clone <repository-url>
cd cs6300-hw5

# Install dependencies (macOS)
make install-mac

# Or install dependencies (Linux/WSL)
make install

# Set up environment variables
# Create .env file and add your GEMINI_API_KEY
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 2. Build Vector Database
```bash
# Index the BoardGameGeek data (creates chroma_db/)
make index
```

### 3. Start Querying
```bash
# Launch interactive query interface
make query

# Or run a quick test
make test-query
```

## 📊 Dataset

The system uses BoardGameGeek data including:
- **21,924 games** with rich metadata
- **Mechanics**: Worker Placement, Cooperative, Deck Building, etc.
- **Themes**: Fantasy, Sci-Fi, Medieval, etc.
- **Categories**: Strategy, Party, Family, etc.
- **Ratings**: Community and Bayes average ratings
- **Player counts** and **playtime** information

## 🔍 Query Examples

The system understands natural language queries:

```
"What are good strategy games for 2 players?"
"Find cooperative games for family game night with 4 people"
"Strategy games rated 8.5 and above"
"Worker placement games with fantasy themes under 90 minutes"
"Best party games for large groups"
```

## 🛠 Architecture

### Core Components

1. **QueryPreprocessor**: LLM-based filter extraction from natural language
2. **BoardGameRAG**: Main RAG pipeline with retrieval and generation
3. **Intelligent Filtering**: Metadata-based filtering with fallbacks
4. **HuggingFace Embeddings**: Local embeddings using all-MiniLM-L6-v2
5. **ChromaDB**: Vector database for semantic search

### Data Flow

```
User Query → LLM Preprocessing → Vector Search → Metadata Filtering → Rating Sort → Deduplication → LLM Response
```

## 📁 Project Structure

```
cs6300-hw5/
├── src/
│   ├── indexing.py          # Data processing and vector database creation
│   ├── rag_agent.py         # Main RAG system with query preprocessing
│   └── __init__.py          # Package initialization
├── data/                    # BoardGameGeek dataset
├── chroma_db/              # Generated vector database (created by indexing)
├── Makefile                # Build automation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🧪 Testing

Test the system with a simple query:

```bash
# Run a quick test query
make test-query
```

For more comprehensive testing, you can run queries interactively:

```bash
# Launch interactive interface and try various queries
make query
```

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# Required: Gemini API key for LLM responses
GEMINI_API_KEY=your_api_key_here

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
```

Create a `.env` file in the project root with these variables.

### System Parameters
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector Database**: ChromaDB with persistence
- **LLM**: Gemini 2.5 Flash
- **Default Results**: Top 20 games per query
- **Indexing Strategy**: 2-chunk per game (description + metadata)

## 🔧 Advanced Usage

### Disable LLM Preprocessing
```bash
# Use keyword-based filtering only
python -c "from src.rag_agent import BoardGameRAG; rag = BoardGameRAG(use_preprocessing=False); rag.query('your question')"
```

### Customize Result Count
```python
from src.rag_agent import BoardGameRAG
rag = BoardGameRAG(top_k=10)  # Return fewer results
```

### Direct Filter Access
```python
# Access extracted filters directly
preprocessor = QueryPreprocessor()
filters = preprocessor.preprocess("cooperative games for 4 players")
print(filters)  # {'mechanics': ['Cooperative Game'], 'min_players': 4, 'max_players': 4}
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GEMINI_API_KEY` is set in `.env`
2. **Import Errors**: Run `make install` to install dependencies
3. **No Results**: Check if `chroma_db/` exists, run `make index` if not
4. **Slow First Run**: HuggingFace model downloads on first use

### Cleaning Up
```bash
# Remove vector database only
make clean-index

# Remove all generated files
make clean

# Remove everything including virtual environment
make clean-all
```

## 📝 Technical Details

### Query Processing Pipeline

1. **Input**: Natural language query
2. **Preprocessing**: LLM extracts structured filters (optional)
3. **Vector Search**: Semantic similarity using embeddings
4. **Filtering**: Apply metadata filters (mechanics, ratings, etc.)
5. **Sorting**: Order by rating when relevant
6. **Deduplication**: Remove duplicate games
7. **Generation**: LLM creates final response

### Filtering Capabilities

- **Mechanics**: Worker Placement, Cooperative, Deck Building, etc.
- **Themes**: Fantasy, Sci-Fi, Medieval, Horror, etc.
- **Categories**: Strategy, Party, Family, Abstract, etc.
- **Player Count**: Exact numbers, ranges, or special cases (solo, large group)
- **Ratings**: Threshold filtering with inclusive/exclusive operators
- **Playtime**: Quick (<30min), long (>120min), or custom ranges

### Performance Optimizations

- **Local Embeddings**: No API rate limits
- **Smart Candidate Retrieval**: 3x-15x multiplier based on filter complexity
- **Graceful Fallbacks**: Relaxes filters when too few results found
- **Efficient Deduplication**: Early termination when target count reached

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- BoardGameGeek for the comprehensive game dataset
- HuggingFace for local embedding models
- LangChain for RAG framework
- ChromaDB for vector storage
- Google Gemini for language model capabilities