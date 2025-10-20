# BoardGameGeek RAG Agent

An AI-powered board game recommendation system that understands natural language queries and finds the perfect games using intelligent semantic search and metadata filtering.

**Key Capabilities:**
- Processes natural language queries with LLM-based preprocessing
- Searches 21,924 BoardGameGeek games using semantic similarity
- Filters by mechanics, themes, player counts, ratings, and playtime
- Generates structured recommendations with detailed metadata
- Provides automated quality evaluation using LLM-as-a-judge

**Query Processing:** Transform "cooperative games for 4 players rated above 8.0" into structured filters, semantic search, and ranked recommendations. The agent prioritizes relevance while respecting user constraints and preferences.

## 🚀 Quick Start

## Prerequisites

- **Python 3.12+**  
  Make sure you are using Python 3.12 or later.

```bash
python3 --version
```

- **API Keys (stored in `.env`)**  
  You will need API keys for the LLM services:

1. **Gemini API key**  
   - Create via [Google AI Studio](https://aistudio.google.com/) → "Get API key"  
   - Save in `.env` as:  
     ```
     GEMINI_API_KEY=your_gemini_api_key
     ```

2. **LangSmith API Key**  
   - Create via [LangSmith](https://smith.langchain.com/)  
   - Save in `.env` as:  
     ```
     LANGCHAIN_API_KEY=your_langsmith_api_key
     ```

## Setup

1. **Create a virtual environment**

Mac:
```bash
make .virtual_environment
source .virtual_environment/bin/activate
```

2. **Install dependencies**

On Mac:
```bash
make install-mac
```

On Linux (teacher's original setup):
```bash
make install
```

---

### Build Vector Database
```bash
make index
```

This processes the BoardGameGeek dataset and creates a searchable vector database with 43,848 document chunks.

### Run the RAG Agent
```bash
make query
```

### Setup Evaluation (Optional)
```bash
make ollama-setup
```

This installs Ollama and Llama 3.2 for automated quality assessment with detailed scoring.

### 💡 Example Usage

```bash
# 1. Build the vector database
make index

# 2. Run the interactive agent
make query

# 3. Try these queries:
# "What are good strategy games for 2 players?"
# "Cooperative games for family game night with 4 people" 
# "Games rated above 8.5 with worker placement"
```

The agent will process your natural language query and return personalized game recommendations.

## Example Queries

Try these natural language queries with the system:

```bash
# After running 'make query', try:
"What are good strategy games for 2 players?"
"Cooperative games for family game night with 4 people" 
"Games rated above 8.5 with worker placement mechanics"
"Quick party games under 30 minutes for large groups"
"Fantasy themed games like Gloomhaven but shorter"
```

## Available Make Commands
```bash
# Environment Setup
make install-mac        # Install on macOS with Homebrew
make install           # Install on Linux/Ubuntu with apt

# Data & Indexing
make explore           # Explore the BoardGameGeek dataset
make index             # Build vector database from BoardGameGeek data

# Running the System
make query             # Interactive board game recommendation agent

# Evaluation Setup (Optional)
make ollama-setup      # Complete Ollama + Llama 3.2 setup
make ollama-install    # Install Ollama only
make ollama-start      # Start Ollama service
make ollama-stop       # Stop Ollama service
make ollama-status     # Check Ollama status

# Maintenance
make clean             # Clean generated files
make clean-index       # Remove vector database only
make clean-all         # Deep clean including virtual environment
```

See `src/rag_agent.py` for complete configuration options.

## 🏗️ Architecture

### 3-Component System

The system uses a streamlined 3-component architecture:

1. **Query Preprocessor** - LLM-based filter extraction and query sanitization
2. **Vector Search Engine** - ChromaDB with HuggingFace embeddings for semantic similarity
3. **Response Generator** - Gemini 2.5 Flash for structured game recommendations

### Core Agent (`src/rag_agent.py`)
- **Google Gemini Integration**: Uses Gemini 2.5 Flash with LangChain
- **Query Sanitization**: Optimizes queries for better semantic search
- **Intelligent Filtering**: Metadata-based filtering with graceful fallbacks
- **Local Embeddings**: HuggingFace all-MiniLM-L6-v2 (no API limits)
- **Automated Evaluation**: Ollama Llama 3.2 LLM-as-a-judge scoring

## 🧪 Testing

#### Interactive Testing
```bash
make query             # Full interactive interface with example queries
```

Note: This project does not include automated unit tests - testing is done through the interactive query interface when the judging system with Ollama is on.

## 📊 Dataset

After running `make index`, you'll have access to:

### **21,924 BoardGameGeek Games** with rich metadata:
- **Mechanics**: Worker Placement, Cooperative, Deck Building, Area Control, Hand Management
- **Themes**: Fantasy, Science Fiction, Medieval, Horror, Economic, War
- **Categories**: Strategy, Party, Family, Abstract, Thematic
- **Ratings**: Average ratings + Bayes-adjusted Geek ratings
- **Player Info**: Min/max players, recommended counts, playtime
- **Complexity**: Weight scores from 1-5

### Generate Query Examples
```bash
# Try these natural language queries:
"What are good strategy games for 2 players?"
"Cooperative games for family game night with 4 people"
"Games rated above 8.5 with worker placement mechanics"
"Quick party games under 30 minutes for large groups"
"Fantasy themed games like Gloomhaven but shorter"
```

## File Structure
```
cs6300-hw5/
│
├── 📋 Core Files
│   ├── README.md          # Project documentation
│   ├── Makefile           # Commands (index, query, ollama-setup, etc.)
│   ├── requirements.txt   # Python dependencies
│   └── .env               # API keys
│
├── 🧠 Source Code (src/)
│   ├── rag_agent.py      # Main RAG agent with query preprocessing
│   ├── indexing.py       # Vector database creation and data processing
│   └── __init__.py       # Package initialization
│
├── �️ Scripts (scripts/)
│   └── explore_data.py   # Dataset exploration and analysis
│
├── �📊 Data
│   ├── data/             # BoardGameGeek dataset (CSV files)
│   └── chroma_db/        # Generated vector database (43,848 chunks)
│
└── ⚙️ Config
    └── .env              # Environment variables (API keys)
```

## 🔧 Advanced Configuration

### Environment Variables (.env)
```bash
# Required: Gemini API key for LLM responses
GEMINI_API_KEY=your_gemini_api_key

# Required: LangSmith API key for tracing and monitoring
LANGCHAIN_API_KEY=your_langsmith_api_key
```

### System Parameters
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions, local)
- **Vector Database**: ChromaDB with persistence
- **LLM**: Gemini 2.5 Flash for query preprocessing and answer generation
- **Evaluation Judge**: Ollama Llama 3.2 (optional, for quality scoring)
- **Default Results**: Top 20 games per query
- **Indexing Strategy**: 2 chunks per game (description + metadata summary)

## Cleaning Up
```bash
# Remove vector database only
make clean-index

# Remove generated files
make clean

# Remove everything including virtual environment
make clean-all
```

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BoardGameGeek**: Comprehensive game dataset and community ratings
- **Google Gemini**: Advanced AI capabilities for query processing and response generation
- **HuggingFace**: Local embedding models (all-MiniLM-L6-v2)
- **ChromaDB**: Efficient vector storage and similarity search
- **Ollama**: Local LLM hosting for automated evaluation
- **LangChain**: Framework for RAG system development