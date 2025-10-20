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
	@echo
	@echo "Evaluation (Ollama):"
	@echo "  ollama-install            - Install Ollama (macOS only)"
	@echo "  ollama-start              - Start Ollama service"
	@echo "  ollama-stop               - Stop Ollama service"
	@echo "  ollama-pull-model         - Download Llama 3.2 model"
	@echo "  ollama-status             - Check Ollama service status"
	@echo "  ollama-setup              - Complete Ollama setup (install + start + model)"
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

clean:
	rm -rf chroma_db/
	rm -rf __pycache__ src/__pycache__
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete

clean-all: clean
	rm -rf $(VENV)

clean-index:
	rm -rf chroma_db/

# Ollama management commands
ollama-install:
	@echo "Installing Ollama (macOS only)..."
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		brew list --versions ollama > /dev/null 2>&1 || brew install ollama; \
		echo "✓ Ollama installed"; \
	else \
		echo "❌ This command is for macOS only. For Linux, run:"; \
		echo "   curl -fsSL https://ollama.ai/install.sh | sh"; \
	fi

ollama-start:
	@echo "Starting Ollama service..."
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		brew services start ollama; \
		echo "✓ Ollama service started"; \
	else \
		echo "Starting Ollama manually..."; \
		ollama serve & \
		echo "✓ Ollama service started in background"; \
	fi

ollama-stop:
	@echo "Stopping Ollama service..."
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		brew services stop ollama; \
		echo "✓ Ollama service stopped"; \
	else \
		pkill -f "ollama serve" || echo "Ollama not running"; \
		echo "✓ Ollama service stopped"; \
	fi

ollama-pull-model:
	@echo "Downloading Llama 3.2 model (this may take a few minutes)..."
	ollama pull llama3.2
	@echo "✓ Llama 3.2 model downloaded"

ollama-status:
	@echo "Checking Ollama status..."
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		brew services list | grep ollama || echo "Ollama not managed by brew services"; \
	fi
	@ollama list 2>/dev/null || echo "❌ Ollama not running or not installed"

ollama-setup: ollama-install ollama-start ollama-pull-model
	@echo "✅ Ollama setup complete!"
	@echo "   • Ollama installed and running"
	@echo "   • Llama 3.2 model downloaded"
	@echo "   • Ready for RAG evaluation"