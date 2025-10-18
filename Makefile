VENV := .virtual_environment

all: help

help:
	@echo
	@echo "BoardGameGeek RAG System - Available Commands:"
	@echo "=============================================="
	@echo "Setup:"
	@echo "  install                   - Install all dependencies (cross-platform)"
	@echo "  install-mac               - Install dependencies on macOS (uses brew)"
	@echo "  install-pip               - Install Python packages only"
	@echo
	@echo "Data & Indexing:"
	@echo "  explore                   - Explore the BoardGameGeek dataset"
	@echo "  index                     - Build vector database from BGG data"
	@echo "  clean-index               - Remove vector database"
	@echo
	@echo "RAG System:"
	@echo "  query                     - Interactive RAG query interface"
	@echo "  test-query                - Run a simple test query"
	@echo
	@echo "Environment:"
	@echo "  clean                     - Clean all generated files"
	@echo "  clean-all                 - Clean everything including venv"
	@echo

$(VENV):
	python3.12 -m venv $(VENV)

install: install-deb install-pip

install-deb:
	@echo python3.12-venv is necessary for venv.
	@echo ffmpeg is necessary to read audio files for ASR
	for package in python3.12-venv ffmpeg; do \
		dpkg -l | egrep '^ii *'$${package}' ' 2>&1 > /dev/null || sudo apt install $${package}; \
	done

install-pip: $(VENV)
	source $(VENV)/bin/activate; pip3 install --upgrade -r requirements.txt

install-mac: install-deb-mac install-pip
	
install-deb-mac:
	@echo python@3.12 is necessary for venv.
	@echo ffmpeg is necessary to read audio files for ASR
	for package in python@3.12 ffmpeg; do \
		brew list --versions $${package} 2>&1 > /dev/null || brew install $${package}; \
	done

explore:
	source $(VENV)/bin/activate; python -m scripts.explore_data

index:
	source $(VENV)/bin/activate; python -m src.indexing

query:
	source $(VENV)/bin/activate; python -m src.rag_agent

test-query:
	source $(VENV)/bin/activate; python -c "from src.rag_agent import BoardGameRAG; rag = BoardGameRAG(use_preprocessing=False); rag.query('What are good cooperative games?')"

clean:
	rm -rf chroma_db/
	rm -rf __pycache__ src/__pycache__
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete

clean-all: clean
	rm -rf $(VENV)

clean-index:
	rm -rf chroma_db/