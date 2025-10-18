VENV := .virtual_environment

all: help

help:
	@echo
	@echo "Targets:"
	@echo "install                     - Install environment necessary to support this project."
	@echo "install-deb                 - Install OS packages necessary to support this project. Assumes apt/dpkg package management system."
	@echo "install-pip                 - Install Python pakcages necessary to suport this project."
	@echo "code-agent-gemini-demo      - Run the demo CodeAgent using the Gemini API."
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
	source $(VENV)/bin/activate; python -c "from src.rag_agent import BoardGameRAG; rag = BoardGameRAG(); rag.query('What are good cooperative games?')"

clean-index:
	rm -rf chroma_db/